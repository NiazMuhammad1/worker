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
    source_language: str = "en"


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
        raise Exception(f"No Argos package found for {from_code} to {to_code}")

    package_path = package_to_install.download()
    argostranslate.package.install_from_path(package_path)


def translate_text(text: str, from_code: str = "en", to_code: str = "ar") -> str:
    if not text.strip():
        return ""

    return argostranslate.translate.translate(text, from_code, to_code)


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
                file.write(f"<c.arabic>{text}</c>\n\n")
            else:
                file.write(f"{text}\n\n")


def process_video(
    video_id: str,
    gcs_video_path: str,
    callback_url: Optional[str],
    source_language: str
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
        print(f"[JOB {video_id}] Checking Argos EN to AR package", flush=True)
        install_argos_language_package("en", "ar")

        print(f"[JOB {video_id}] Downloading video from GCS: {gcs_video_path}", flush=True)
        download_from_gcs(gcs_video_path, local_video)

        print(f"[JOB {video_id}] Extracting audio with FFmpeg", flush=True)
        extract_audio(local_video, local_audio)

        print(f"[JOB {video_id}] Running Whisper transcription", flush=True)
        whisper_segments, info = model.transcribe(
            local_audio,
            language=source_language,
            beam_size=5,
            vad_filter=True
        )

        english_segments = []

        for segment in whisper_segments:
            english_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        print(f"[JOB {video_id}] English segments created: {len(english_segments)}", flush=True)

        arabic_segments = []

        print(f"[JOB {video_id}] Translating English subtitles to Arabic", flush=True)

        for segment in english_segments:
            arabic_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": translate_text(segment["text"], "en", "ar")
            })

        print(f"[JOB {video_id}] Writing VTT files", flush=True)

        write_vtt(english_segments, english_vtt)
        write_vtt(arabic_segments, arabic_vtt, rtl=True)

        subtitle_en_gcs_path = f"subtitles/{video_id}/subtitles_en.vtt"
        subtitle_ar_gcs_path = f"subtitles/{video_id}/subtitles_ar.vtt"

        print(f"[JOB {video_id}] Uploading English VTT to GCS: {subtitle_en_gcs_path}", flush=True)
        upload_to_gcs(english_vtt, subtitle_en_gcs_path)

        print(f"[JOB {video_id}] Uploading Arabic VTT to GCS: {subtitle_ar_gcs_path}", flush=True)
        upload_to_gcs(arabic_vtt, subtitle_ar_gcs_path)

        payload = {
            "video_id": video_id,
            "status": "completed",
            "subtitle_en_path": subtitle_en_gcs_path,
            "subtitle_ar_path": subtitle_ar_gcs_path
        }

        if callback_url:
            print(f"[JOB {video_id}] Sending callback to Laravel", flush=True)
            requests.post(callback_url, json=payload, timeout=30)

        print(f"[JOB {video_id}] Subtitle job completed successfully", flush=True)

    except Exception as error:
        print(f"[JOB {video_id}] ERROR: {str(error)}", flush=True)

        if callback_url:
            requests.post(callback_url, json={
                "video_id": video_id,
                "status": "failed",
                "error": str(error)
            }, timeout=30)

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
            print(f"[JOB {video_id}] Temporary files cleaned", flush=True)


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