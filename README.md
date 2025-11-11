---
title: FastAPI
description: A FastAPI server
tags:
  - fastapi
  - hypercorn
  - python
---

# UN Webcast Transcription Pipeline

This service ingests Kaltura recordings from the UN Webcast platform, downloads them
with `yt-dlp`, remuxes the audio into 20&nbsp;MB chunks via `ffmpeg`, sends each chunk to
the OpenAI Whisper API, and stores both the metadata and transcripts for later
retrieval. All major steps emit structured events through [Logfire](https://logfire.pydantic.dev/)
so you can observe the pipeline in real time.

## ✨ Features

- Elegant HTML dashboard on `/` to queue and review webcast meetings
- FastAPI-powered REST API to submit and monitor webcast meetings
- Automatic downloading and chunking of Kaltura recordings via `yt-dlp` and `ffmpeg`
- Configurable Whisper API transcription pipeline
- Structured logging with Logfire across the entire workflow
- SQLite persistence for meetings and transcript segments

## 🚀 Getting Started

- Clone locally and install packages with pip using `pip install -r requirements.txt`
- Run locally using `hypercorn main:app --reload`
- Install [FFmpeg](https://ffmpeg.org/download.html) so `pydub` can re-encode audio chunks when calling the utilities in `services.py`.

## 🔊 Audio chunking

The `split_file` helper in `services.py` remuxes or re-encodes large audio files
into ≤20&nbsp;MB chunks using `pydub`/FFmpeg to ensure each exported segment is
playable on its own. Install FFmpeg before using this helper locally or in CI
environments.

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Provide credentials**

   - Install `ffmpeg` on the host so the service can extract audio chunks.
   - Set the `OPENAI_API_KEY` environment variable so the Whisper API can be invoked.
   - Optionally set `WHISPER_API_MODEL` (defaults to `whisper-1`) and
     `AUDIO_CHUNK_BITRATE` (defaults to `128000`).
   - To forward logs to Logfire Cloud, set `LOGFIRE_SEND=true` and configure the
     relevant environment variables per the Logfire documentation.

3. **Run the API**

   ```bash
   hypercorn main:app --reload
   ```

4. **Use the web dashboard**

   Navigate to `http://localhost:8000/` to submit a webcast URL and monitor progress in
   real time from the built-in interface.

5. **Submit via the API**

   Prefer to script it? Use the interactive docs at `http://localhost:8000/docs` or send
   a POST request to `/meetings` with JSON similar to:

   ```json
   {
     "title": "Security Council Briefing",
     "source_url": "https://vod.unwebtv.org/asset/kaltura_id"
   }
   ```

6. **Run the pipeline from the CLI**

   To test the workflow end-to-end without running a server, use the helper script:

   ```bash
   python cli.py --title "Security Council Briefing" \
     https://webtv.un.org/en/asset/k1w/k1wp6tpfo2
   ```

   This downloads the webcast, extracts playable MP3 chunks, submits them to the
   Whisper API, and stores the results exactly as the API would.

## 📦 Data Storage

- Recordings and chunked files are stored under `data/meetings/<meeting_id>/`
- Metadata and transcripts are stored in `data/meetings.db`

## 📝 Notes

- Ensure `ffmpeg` is installed and on the PATH for `yt-dlp` and audio chunking to work.
- Whisper API usage incurs OpenAI charges—review pricing and monitor usage.
