from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

try:
    import logfire
except ImportError:  # pragma: no cover - fallback for environments without logfire
    logfire = None

from openai import OpenAI
from sqlalchemy.orm import Session
from yt_dlp import YoutubeDL

from database import SessionLocal
from models import Meeting, TranscriptSegment

logger = logging.getLogger(__name__)

CHUNK_SIZE_BYTES = 20 * 1024 * 1024
DEFAULT_AUDIO_BITRATE = int(os.getenv("AUDIO_CHUNK_BITRATE", "128000"))
WHISPER_API_MODEL = os.getenv("WHISPER_API_MODEL", "whisper-1")


def _log(level: str, message: str, **kwargs: object) -> None:
    if logfire is not None:
        log_method = getattr(logfire, level, None)
        if callable(log_method):
            log_method(message, **kwargs)

    log_method = getattr(logger, level if level != "exception" else "error")
    if kwargs:
        log_method("%s | %s", message, kwargs)
    else:
        log_method(message)


def download_recording(source_url: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    _log("info", "Starting download", source_url=source_url, target_dir=str(target_dir))
    ydl_opts = {
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source_url, download=True)
        filename = ydl.prepare_filename(info)

    downloaded_path = Path(filename)
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Downloaded file not found at {downloaded_path}")

    _log("info", "Download completed", file=str(downloaded_path))

    return downloaded_path


def split_file(path: Path, chunk_size: int = CHUNK_SIZE_BYTES) -> List[Path]:
    chunk_dir = path.parent / "chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    segment_seconds = max(1, math.floor(chunk_size * 8 * 0.9 / DEFAULT_AUDIO_BITRATE))
    attempt = 0
    while True:
        attempt += 1
        for item in chunk_dir.glob("*"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        output_pattern = chunk_dir / "chunk_%04d.mp3"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-b:a",
            f"{DEFAULT_AUDIO_BITRATE // 1000}k",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ]

        _log(
            "info",
            "Creating audio chunks",
            command=" ".join(command),
            attempt=attempt,
            segment_seconds=segment_seconds,
        )
        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # noqa: PERF203
            _log("exception", "Chunking failed", error=str(exc))
            raise

        chunk_paths = sorted(chunk_dir.glob("chunk_*.mp3"))
        if not chunk_paths:
            raise RuntimeError("No chunks were produced by ffmpeg")

        oversized = [p for p in chunk_paths if p.stat().st_size > chunk_size]
        if not oversized or segment_seconds == 1:
            _log(
                "info",
                "Chunking completed",
                chunk_count=len(chunk_paths),
                chunk_dir=str(chunk_dir),
            )
            return chunk_paths

        segment_seconds = max(1, segment_seconds // 2)
        _log(
            "warning",
            "Resplitting due to oversized chunks",
            oversized_count=len(oversized),
            new_segment_seconds=segment_seconds,
        )


_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _log("info", "Initializing OpenAI client", model=WHISPER_API_MODEL)
        _openai_client = OpenAI()
    return _openai_client


def transcribe_chunks(chunks: Iterable[Path]) -> List[str]:
    client = get_openai_client()
    transcripts: List[str] = []
    for chunk in chunks:
        _log("info", "Submitting chunk for transcription", chunk=str(chunk))
        try:
            with chunk.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=WHISPER_API_MODEL,
                    file=audio_file,
                )
        except Exception as exc:  # noqa: BLE001
            _log("exception", "Transcription failed", chunk=str(chunk), error=str(exc))
            raise
        text = (response.text or "").strip()
        transcripts.append(text)
        _log(
            "info",
            "Received transcription",
            chunk=str(chunk),
            characters=len(text),
        )
    return transcripts


def process_meeting(meeting: Meeting, session: Session) -> None:
    _log("info", "Processing started", meeting_id=meeting.id)
    session.add(meeting)
    session.commit()

    try:
        meeting.status = "downloading"
        session.commit()
        _log("info", "Status updated", meeting_id=meeting.id, status=meeting.status)

        with tempfile.TemporaryDirectory(prefix=f"meeting_{meeting.id}_") as temp_dir:
            temp_path = Path(temp_dir)
            recording_path = download_recording(meeting.source_url, temp_path)
            stored_dir = Path("data") / "meetings" / str(meeting.id)
            stored_dir.mkdir(parents=True, exist_ok=True)
            stored_path = stored_dir / recording_path.name
            shutil.copy2(recording_path, stored_path)
            meeting.recording_path = str(stored_path)
            session.commit()
            _log(
                "info",
                "Recording stored",
                meeting_id=meeting.id,
                path=meeting.recording_path,
            )

            meeting.status = "chunking"
            session.commit()
            _log("info", "Status updated", meeting_id=meeting.id, status=meeting.status)
            chunk_paths = split_file(stored_path)

            meeting.status = "transcribing"
            session.commit()
            _log("info", "Status updated", meeting_id=meeting.id, status=meeting.status)
            transcripts = transcribe_chunks(chunk_paths)

        meeting.status = "saving"
        meeting.transcript = "\n".join(transcripts)
        meeting.completed_at = datetime.utcnow()
        session.commit()
        _log("info", "Transcript stored", meeting_id=meeting.id, length=len(meeting.transcript or ""))

        for index, (chunk_path, text) in enumerate(zip(chunk_paths, transcripts)):
            segment = TranscriptSegment(
                meeting_id=meeting.id,
                chunk_index=index,
                chunk_path=str(chunk_path),
                text=text,
            )
            session.add(segment)

        session.commit()

        meeting.status = "completed"
        session.commit()
        _log("info", "Processing completed", meeting_id=meeting.id)
    except Exception as exc:  # noqa: BLE001
        _log("exception", "Processing failed", meeting_id=meeting.id, error=str(exc))
        session.rollback()
        meeting.status = "failed"
        meeting.error_message = str(exc)
        session.commit()
        raise


def process_meeting_task(meeting_id: int) -> None:
    session = SessionLocal()
    try:
        meeting = session.get(Meeting, meeting_id)
        if meeting is None:
            _log("warning", "Meeting not found", meeting_id=meeting_id)
            return
        process_meeting(meeting, session)
    finally:
        session.close()
