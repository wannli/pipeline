from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, HttpUrl


class TranscriptSegmentRead(BaseModel):
    id: int
    chunk_index: int
    chunk_path: str
    text: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


class MeetingCreate(BaseModel):
    title: str
    source_url: HttpUrl


class MeetingRead(BaseModel):
    id: int
    title: str
    source_url: HttpUrl
    status: str
    transcript: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    class Config:
        orm_mode = True


class MeetingDetail(MeetingRead):
    segments: List[TranscriptSegmentRead]
