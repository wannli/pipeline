from __future__ import annotations

import argparse
import logging
import os

try:
    import logfire
except ImportError:  # pragma: no cover - fallback when dependency missing
    logfire = None

from database import Base, engine, SessionLocal
from models import Meeting
from services import process_meeting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if logfire is not None:
    send_to_logfire = os.getenv("LOGFIRE_SEND", "false").lower() in {"1", "true", "yes"}
    logfire.configure(send_to_logfire=send_to_logfire)


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


def run_pipeline(title: str, source_url: str) -> None:
    """Create a meeting entry and run the processing pipeline synchronously."""
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        meeting = Meeting(
            title=title,
            source_url=source_url,
            status="queued",
        )
        session.add(meeting)
        session.commit()
        session.refresh(meeting)

        _log("info", "Starting synchronous processing", meeting_id=meeting.id)
        process_meeting(meeting, session)
        _log("info", "Finished processing", meeting_id=meeting.id)
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the webcast pipeline once")
    parser.add_argument(
        "source_url",
        help="Kaltura webcast URL to download and transcribe",
    )
    parser.add_argument(
        "--title",
        default="UN Webcast Meeting",
        help="Human-friendly meeting title (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args.title, args.source_url)


if __name__ == "__main__":
    main()
