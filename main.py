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

model = None
job_lock = threading.Lock()
load_dotenv()

API_KEY = os.getenv("WORKER_API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
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
    source_language: Optional[str] = None


def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def install_argos_language_package(from_code: str = "en", to_code: str = "ar"):
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


def translate_text(text: str, from_code: str = "en", to_code: str = "ar") -> str:
    if not text.strip():
        return ""

    try:
        return argostranslate.translate.translate(text, from_code, to_code)
    except Exception as error:
        print(f"Translation error: {error}", flush=True)
        return text


def download_from_gcs(gcs_path: str, local_path: str):
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)


def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    blob = bucket.blob(gcs_path)

    content_type = "text/vtt" if gcs_path.endswith(".vtt") else None

    blob.upload_from_filename(
        local_path,
        content_type=content_type
    )

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


def run_whisper_transcription(
    audio_path: str,
    language: Optional[str] = None,
    task: str = "transcribe"
):
    options = {
        "beam_size": 8,
        "vad_filter": True,
        "condition_on_previous_text": True,
        "task": task,
    }

    if language:
        options["language"] = language

    whisper_segments, info = model.transcribe(
        audio_path,
        **options
    )

    segments = []

    for segment in whisper_segments:
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return segments, info.language


def send_callback(callback_url: Optional[str], payload: dict, video_id: str):
    if not callback_url:
        print(f"[JOB {video_id}] No callback URL provided", flush=True)
        return

    try:
        print(f"[JOB {video_id}] Sending callback to Laravel", flush=True)

        response = requests.post(
            callback_url,
            json=payload,
            timeout=30,
            allow_redirects=False
        )

        print(f"[JOB {video_id}] Callback status: {response.status_code}", flush=True)
        print(f"[JOB {video_id}] Callback response preview: {response.text[:500]}", flush=True)

        if response.status_code in [301, 302, 307, 308]:
            redirect_url = response.headers.get("Location")

            print(f"[JOB {video_id}] Callback redirected to: {redirect_url}", flush=True)

            if redirect_url:
                redirect_response = requests.post(
                    redirect_url,
                    json=payload,
                    timeout=30,
                    allow_redirects=False
                )

                print(
                    f"[JOB {video_id}] Redirect callback status: {redirect_response.status_code}",
                    flush=True
                )

                print(
                    f"[JOB {video_id}] Redirect callback response preview: {redirect_response.text[:500]}",
                    flush=True
                )

    except Exception as callback_error:
        print(f"[JOB {video_id}] Callback failed: {callback_error}", flush=True)


def process_video(
    video_id: str,
    gcs_video_path: str,
    callback_url: Optional[str],
    source_language: Optional[str] = None
):

    # =========================================================
    # PREVENT MULTIPLE WHISPER JOBS
    # =========================================================

    if not job_lock.acquire(blocking=False):

        print(f"[JOB {video_id}] Worker busy. Another subtitle job is running.", flush=True)

        fail_payload = {
            "video_id": video_id,
            "status": "failed",
            "error": "Subtitle worker busy. Please retry later."
        }

        send_callback(callback_url, fail_payload, video_id)

        return

    try:

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

            print(f"[JOB {video_id}] Downloading video from GCS: {gcs_video_path}", flush=True)

            download_from_gcs(gcs_video_path, local_video)

            print(f"[JOB {video_id}] Extracting audio with FFmpeg", flush=True)

            extract_audio(local_video, local_audio)

            print(f"[JOB {video_id}] Running Whisper original transcription", flush=True)

            original_segments, detected_language = run_whisper_transcription(
                local_audio,
                language=source_language,
                task="transcribe"
            )

            detected_language = source_language or detected_language

            print(f"[JOB {video_id}] Detected/source language: {detected_language}", flush=True)

            print(f"[JOB {video_id}] Original segments created: {len(original_segments)}", flush=True)

            english_segments = []
            arabic_segments = []

            # =========================================================
            # NON-ENGLISH VIDEOS
            # =========================================================

            if detected_language != "en":

                print(f"[JOB {video_id}] Non-English video detected", flush=True)

                arabic_segments = original_segments

                print(f"[JOB {video_id}] Running Whisper English translation", flush=True)

                english_segments, _ = run_whisper_transcription(
                    local_audio,
                    language=detected_language,
                    task="translate"
                )

            # =========================================================
            # ENGLISH VIDEOS
            # =========================================================

            else:

                print(f"[JOB {video_id}] English video detected", flush=True)

                english_segments = original_segments

                print(f"[JOB {video_id}] Installing Argos EN -> AR package", flush=True)

                install_argos_language_package("en", "ar")

                print(f"[JOB {video_id}] Translating English subtitles to Arabic using Argos", flush=True)

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

            subtitle_en_gcs_path = f"subtitles/{video_id}/subtitles_en.vtt"

            subtitle_ar_gcs_path = f"subtitles/{video_id}/subtitles_ar.vtt"

            print(f"[JOB {video_id}] Uploading English VTT to GCS", flush=True)

            upload_to_gcs(
                english_vtt,
                subtitle_en_gcs_path
            )

            print(f"[JOB {video_id}] Uploading Arabic/original VTT to GCS", flush=True)

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

            send_callback(callback_url, payload, video_id)

            print(f"[JOB {video_id}] Subtitle job completed successfully", flush=True)

        except Exception as error:

            print(f"[JOB {video_id}] ERROR: {str(error)}", flush=True)

            fail_payload = {
                "video_id": video_id,
                "status": "failed",
                "error": str(error)
            }

            send_callback(callback_url, fail_payload, video_id)

        finally:

            if work_dir.exists():

                shutil.rmtree(work_dir)

                print(f"[JOB {video_id}] Temporary files cleaned", flush=True)

    finally:

        # =========================================================
        # RELEASE LOCK
        # =========================================================

        job_lock.release()

        print(f"[JOB {video_id}] Worker lock released", flush=True)


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
        request.callback_url,
        request.source_language
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