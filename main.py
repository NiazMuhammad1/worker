import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel
from google.cloud import storage
from faster_whisper import WhisperModel

import argostranslate.package
import argostranslate.translate


load_dotenv()

API_KEY = os.getenv("WORKER_API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

app = FastAPI(title="LMS Subtitle Worker")

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)

model = None


class SubtitleRequest(BaseModel):
    video_id: str
    gcs_video_path: str
    callback_url: Optional[str] = None


def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def install_argos_language_package(from_code: str, to_code: str):
    installed_languages = argostranslate.translate.get_installed_languages()

    for lang in installed_languages:
        if lang.code == from_code:
            for translation in lang.translations_from:
                if translation.to_lang.code == to_code:
                    return

    print(f"Installing Argos package: {from_code} -> {to_code}", flush=True)

    argostranslate.package.update_package_index()

    available_packages = argostranslate.package.get_available_packages()

    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages
        ),
        None
    )

    if package_to_install is None:
        raise Exception(f"No Argos package found for {from_code} -> {to_code}")

    package_path = package_to_install.download()

    argostranslate.package.install_from_path(package_path)

    print(f"Installed Argos package: {from_code} -> {to_code}", flush=True)


def translate_text(text: str, from_code: str, to_code: str) -> str:
    if not text.strip():
        return ""

    try:
        return argostranslate.translate.translate(
            text,
            from_code,
            to_code
        )
    except Exception as error:
        print(f"Translation error: {error}", flush=True)
        return text


def download_from_gcs(gcs_path: str, local_path: str):
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)


def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return gcs_path


def extract_audio(video_path: str, audio_path: str):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path
    ]

    subprocess.run(command, check=True)


def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds % 1) * 1000)
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def write_vtt(segments: List[Dict], output_path: str, rtl: bool = False):

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("WEBVTT\n\n")

        for segment in segments:

            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])

            text = segment["text"].strip()

            file.write(f"{start} --> {end}\n")

            if rtl:
                file.write(f"<c.rtl>{text}</c>\n\n")
            else:
                file.write(f"{text}\n\n")


def process_video(
    video_id: str,
    gcs_video_path: str,
    callback_url: Optional[str]
):

    print("PROCESS VIDEO FUNCTION STARTED", flush=True)

    global model

    if model is None:

        print(f"[JOB {video_id}] Loading Whisper model: {WHISPER_MODEL}", flush=True)

        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )

        print(f"[JOB {video_id}] Whisper model loaded", flush=True)

    job_uuid = str(uuid.uuid4())

    work_dir = Path("tmp") / job_uuid

    work_dir.mkdir(parents=True, exist_ok=True)

    local_video = str(work_dir / "video.mp4")
    local_audio = str(work_dir / "audio.wav")

    english_vtt = str(work_dir / "subtitles_en.vtt")
    arabic_vtt = str(work_dir / "subtitles_ar.vtt")

    try:

        print(f"[JOB {video_id}] Installing Argos language packages", flush=True)

        install_argos_language_package("en", "ar")
        install_argos_language_package("ar", "en")

        install_argos_language_package("ur", "en")
        install_argos_language_package("en", "ur")

        install_argos_language_package("hi", "en")
        install_argos_language_package("en", "hi")

        print(f"[JOB {video_id}] Downloading video from GCS", flush=True)

        download_from_gcs(gcs_video_path, local_video)

        print(f"[JOB {video_id}] Extracting audio with FFmpeg", flush=True)

        extract_audio(local_video, local_audio)

        print(f"[JOB {video_id}] Running Whisper transcription", flush=True)

        whisper_segments, info = model.transcribe(
            local_audio,
            beam_size=8,
            vad_filter=True,
            condition_on_previous_text=True
        )

        detected_language = info.language

        print(
            f"[JOB {video_id}] Detected language: {detected_language}",
            flush=True
        )

        segments = []

        for segment in whisper_segments:

            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        print(
            f"[JOB {video_id}] Segments created: {len(segments)}",
            flush=True
        )

        english_segments = []
        arabic_segments = []

        # =========================================================
        # ARABIC VIDEO
        # =========================================================

        if detected_language == "ar":

            print(f"[JOB {video_id}] Arabic video detected", flush=True)

            arabic_segments = segments

            for segment in arabic_segments:

                english_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translate_text(
                        segment["text"],
                        "ar",
                        "en"
                    )
                })

        # =========================================================
        # ENGLISH VIDEO
        # =========================================================

        elif detected_language == "en":

            print(f"[JOB {video_id}] English video detected", flush=True)

            english_segments = segments

            for segment in english_segments:

                arabic_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translate_text(
                        segment["text"],
                        "en",
                        "ar"
                    )
                })

        # =========================================================
        # URDU VIDEO
        # =========================================================

        elif detected_language == "ur":

            print(f"[JOB {video_id}] Urdu video detected", flush=True)

            arabic_segments = segments

            for segment in segments:

                english_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translate_text(
                        segment["text"],
                        "ur",
                        "en"
                    )
                })

        # =========================================================
        # HINDI VIDEO
        # =========================================================

        elif detected_language == "hi":

            print(f"[JOB {video_id}] Hindi video detected", flush=True)

            arabic_segments = segments

            for segment in segments:

                english_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translate_text(
                        segment["text"],
                        "hi",
                        "en"
                    )
                })

        # =========================================================
        # FALLBACK
        # =========================================================

        else:

            print(
                f"[JOB {video_id}] Unknown language fallback",
                flush=True
            )

            english_segments = segments

            for segment in english_segments:

                arabic_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translate_text(
                        segment["text"],
                        "en",
                        "ar"
                    )
                })

        print(f"[JOB {video_id}] Writing VTT files", flush=True)

        write_vtt(english_segments, english_vtt)

        write_vtt(
            arabic_segments,
            arabic_vtt,
            rtl=True
        )

        subtitle_en_gcs_path = (
            f"subtitles/{video_id}/subtitles_en.vtt"
        )

        subtitle_ar_gcs_path = (
            f"subtitles/{video_id}/subtitles_ar.vtt"
        )

        print(
            f"[JOB {video_id}] Uploading English VTT",
            flush=True
        )

        upload_to_gcs(
            english_vtt,
            subtitle_en_gcs_path
        )

        print(
            f"[JOB {video_id}] Uploading Arabic VTT",
            flush=True
        )

        upload_to_gcs(
            arabic_vtt,
            subtitle_ar_gcs_path
        )

        payload = {
            "video_id": video_id,
            "status": "completed",
            "subtitle_en_path": subtitle_en_gcs_path,
            "subtitle_ar_path": subtitle_ar_gcs_path,
            "detected_language": detected_language
        }

        if callback_url:

            print(
                f"[JOB {video_id}] Sending callback to Laravel",
                flush=True
            )

            response = requests.post(
                callback_url,
                json=payload,
                timeout=30
            )

            print(
                f"[JOB {video_id}] Callback status: {response.status_code}",
                flush=True
            )

        print(
            f"[JOB {video_id}] Subtitle job completed successfully",
            flush=True
        )

    except Exception as error:

        print(f"[JOB {video_id}] ERROR: {str(error)}", flush=True)

        if callback_url:

            try:

                requests.post(
                    callback_url,
                    json={
                        "video_id": video_id,
                        "status": "failed",
                        "error": str(error)
                    },
                    timeout=30
                )

            except Exception as callback_error:

                print(
                    f"[JOB {video_id}] Callback failed: {callback_error}",
                    flush=True
                )

    finally:

        if work_dir.exists():

            shutil.rmtree(work_dir)

            print(
                f"[JOB {video_id}] Temporary files cleaned",
                flush=True
            )


@app.post("/generate-subtitles")
def generate_subtitles(
    request: SubtitleRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default="")
):

    verify_api_key(x_api_key)

    background_tasks.add_task(
        process_video,
        request.video_id,
        request.gcs_video_path,
        request.callback_url
    )

    return {
        "message": "Subtitle generation started",
        "video_id": request.video_id
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL
    }