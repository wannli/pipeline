from __future__ import annotations

import logging
import os
from typing import List

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy.orm import selectinload

try:
    import logfire
except ImportError:  # pragma: no cover - fallback when dependency missing
    logfire = None

from database import Base, engine, get_session
from models import Meeting
from schemas import MeetingCreate, MeetingDetail, MeetingRead
from services import process_meeting_task

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


app = FastAPI(title="UN Webcast Pipeline")
templates = Jinja2Templates(directory="templates")

if logfire is not None and hasattr(logfire, "instrument_fastapi"):
    logfire.instrument_fastapi(app)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    _log("info", "Database initialized")


@app.get("/", response_class=HTMLResponse)
def root(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    meetings = (
        session.query(Meeting)
        .options(selectinload(Meeting.segments))
        .order_by(Meeting.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "meetings": meetings,
        },
    )


@app.post("/meetings", response_model=MeetingRead, status_code=status.HTTP_201_CREATED)
def create_meeting(
    meeting_in: MeetingCreate,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> Meeting:
    meeting = Meeting(
        title=meeting_in.title,
        source_url=str(meeting_in.source_url),
        status="queued",
    )
    session.add(meeting)
    session.commit()
    session.refresh(meeting)

    background_tasks.add_task(process_meeting_task, meeting.id)
    _log("info", "Meeting queued", meeting_id=meeting.id)

    return meeting


@app.get("/meetings", response_model=List[MeetingRead])
def list_meetings(session: Session = Depends(get_session)) -> List[Meeting]:
    meetings = session.query(Meeting).order_by(Meeting.created_at.desc()).all()
    return meetings


@app.get("/meetings/{meeting_id}", response_model=MeetingDetail)
def get_meeting(meeting_id: int, session: Session = Depends(get_session)) -> Meeting:
    meeting = session.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")
    return meeting
