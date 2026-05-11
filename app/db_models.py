from typing import List, Optional
from beanie import Document, Indexed
from pydantic import EmailStr


class ResumeDocument(Document):
    name: str = ""
    email: Optional[Indexed(EmailStr, unique=True)] = None
    education: str = ""
    skills: List[str] = []

    class Settings:
        name = "profiles"
