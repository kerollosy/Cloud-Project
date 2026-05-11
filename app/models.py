from pydantic import BaseModel, EmailStr
from typing import List, Optional


class ResumeData(BaseModel):
    name: str = ""
    email: Optional[EmailStr] = None
    skills: List[str] = []
    education: List[str] = []


class ExtractionResponse(BaseModel):
    status: str
    message: str
    data: ResumeData | None = None