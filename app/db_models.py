from typing import List, Optional
from datetime import datetime
from enum import Enum
from beanie import Document, Indexed
from pydantic import EmailStr, Field, BaseModel


class ReviewStatus(str, Enum):
    AUTO_APPROVED = "auto-approved"
    PENDING_REVIEW = "pending-review"
    REVIEWED = "reviewed"


class AuditAction(str, Enum):
    EXTRACTED = "extracted"
    AUTO_APPROVED = "auto-approved"
    FLAGGED_FOR_REVIEW = "flagged-for-review"
    APPROVED_BY_HUMAN = "approved-by-human"
    CORRECTED_BY_HUMAN = "corrected-by-human"
    REJECTED = "rejected"


class ConfidenceScores(BaseModel):
    name: float = 0.0
    email: float = 0.0
    education: float = 0.0
    skills: float = 0.85
    overall: float = 0.0


class ResumeDocument(Document):
    name: str = ""
    email: Optional[Indexed(EmailStr, unique=True)] = None
    education: str = ""
    skills: List[str] = []
    status: str = ReviewStatus.AUTO_APPROVED.value  # Store as string value
    confidence: dict = Field(default_factory=dict)  # Store as dict for flexibility
    audit_trail: List[dict] = Field(default_factory=list)

    class Settings:
        name = "profiles"
