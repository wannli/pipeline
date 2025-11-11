"""Audio processing helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

from pydub import AudioSegment


def _derive_chunk_path(source: Path, index: int, extension: str) -> Path:
    """Return a deterministic path for a chunk file."""

    return source.with_name(f"{source.stem}_chunk_{index:04d}.{extension}")


def _encoded_size(segment: AudioSegment, export_format: str) -> int:
    """Return the encoded size of ``segment`` in bytes."""

    buffer = BytesIO()
    segment.export(buffer, format=export_format)
    return buffer.tell()


def _export_chunk(
    chunk: AudioSegment, destination: Path, export_format: str
) -> Path:
    """Export a chunk to ``destination`` and return the path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    chunk.export(destination, format=export_format)
    return destination


def split_file(file_path: str | Path, chunk_size_mb: int = 20) -> List[str]:
    """Split an audio file into playable chunks.

    Args:
        file_path: Path to the source audio file.
        chunk_size_mb: Maximum size per chunk in megabytes.

    Returns:
        A list of file paths pointing to the newly created chunk files.
    """

    source_path = Path(file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if not source_path.is_file():
        raise ValueError(f"Expected a file, received: {file_path}")

    if chunk_size_mb <= 0:
        raise ValueError("`chunk_size_mb` must be a positive number")

    # ``pydub`` uses FFmpeg under the hood, enabling reliable remuxing to the
    # original format.  ``suffix`` includes the dot, so strip it when passed to
    # ``AudioSegment.export``.
    export_format = source_path.suffix.lstrip(".") or "wav"
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    audio = AudioSegment.from_file(source_path)
    if len(audio) == 0:
        raise ValueError("Cannot split an empty audio file")

    # Estimate duration per chunk based on the raw data rate.  ``frame_width``
    # reports the number of bytes per sample for all channels combined.
    bytes_per_second = max(audio.frame_rate * audio.frame_width, 1)
    ms_per_chunk = max(int(chunk_size_bytes / bytes_per_second * 1000), 1)

    chunk_paths: List[str] = []
    start_ms = 0
    index = 0
    while start_ms < len(audio):
        end_ms = min(start_ms + ms_per_chunk, len(audio))
        chunk = audio[start_ms:end_ms]

        encoded_size = _encoded_size(chunk, export_format)
        while encoded_size > chunk_size_bytes and len(chunk) > 1:
            # Shrink the chunk proportionally to the overage.  The ``max`` call
            # guarantees progress toward a smaller segment while keeping the
            # duration positive.
            proportional_length = int(len(chunk) * chunk_size_bytes / encoded_size)
            new_length = max(min(proportional_length, len(chunk) - 1), 1)
            if new_length == len(chunk):
                new_length = len(chunk) - 1
            chunk = chunk[:new_length]
            encoded_size = _encoded_size(chunk, export_format)

        if encoded_size > chunk_size_bytes:
            raise ValueError(
                "Unable to split the audio within the requested size limit. "
                "Try increasing `chunk_size_mb`."
            )

        chunk_path = _derive_chunk_path(source_path, index, export_format)
        _export_chunk(chunk, chunk_path, export_format)
        chunk_paths.append(str(chunk_path))

        start_ms += len(chunk)
        index += 1

    return chunk_paths
