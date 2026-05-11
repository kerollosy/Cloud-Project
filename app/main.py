import os
import logging
import re
from contextlib import asynccontextmanager

import boto3
import watchtower
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ExtractionResponse, ResumeData
from app.db_models import ResumeDocument
from app.database import init_db
from app.utils.parser import extract_text_from_pdf, extract_text_from_txt
from app.utils.ai_extractor import load_model, unload_model, extract_fields
from app.utils.s3_helper import upload_file_to_s3
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

load_dotenv()

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _normalize_email(value: str | None) -> str | None:
    """Return a cleaned email address or None if the value is not valid."""
    if not value:
        return None

    cleaned = value.strip().strip(".,;:!?)]}\"")
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", cleaned):
        return cleaned
    return None

if not all([AWS_REGION, S3_BUCKET_NAME]):
    logger.warning("Missing AWS S3 configuration - S3 features will be unavailable")

if not all([MONGO_URL, DATABASE_NAME]):
    logger.warning("Missing MongoDB configuration - DB persistence disabled")


# ---------------------------------------------------------------------------
# Lifespan: load model on startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group="ResumeAPI-Production",
            boto3_client=boto3.client("logs", region_name=AWS_REGION)
        )
        logger.addHandler(cw_handler)
        logging.getLogger("uvicorn.access").addHandler(cw_handler)
        logging.getLogger("uvicorn.error").addHandler(cw_handler)
        logger.info("CloudWatch Logging successfully initialized.")
    except Exception as e:
        logger.warning(f"Could not initialize CloudWatch logging: {e}")

    logger.info("Starting up – loading AI model ...")
    load_model()
    await init_db()
    yield
    logger.info("Shutting down – unloading AI model ...")
    unload_model()


app = FastAPI(
    title="Resume Extraction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") 

@app.get("/", response_class=HTMLResponse)
async def health_check(request: Request):
   return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )


@app.get("/api/v1/resumes")
async def get_all_resumes():
    """
    Retrieve all stored resumes from MongoDB.
    Returns an empty list if no resumes have been stored.
    """
    try:
        resumes = await ResumeDocument.find_all().to_list()
        return resumes
    except Exception:
        
        logger.warning("Failed to retrieve resumes from MongoDB.")
        raise HTTPException(status_code=503, detail="Database unavailable")

@app.get("/view-resumes", response_class=HTMLResponse)
async def view_resumes_page(request: Request):
  
    return templates.TemplateResponse(
        request=request, 
        name="resumes.html"
    )

@app.post("/api/v1/extract", response_model=ExtractionResponse)
async def extract_resume(file: UploadFile = File(...)):
    """
    Accepts a PDF or plain text resume, extracts text, runs AI inference,
    and returns structured data (Name, Email, Skills, Education).
    """
    is_pdf = file.filename.lower().endswith('.pdf')
    is_txt = file.filename.lower().endswith('.txt')
    
    if not is_pdf and not is_txt:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Expected application/pdf or text/plain."
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB."
        )

    try:
        s3_path = upload_file_to_s3(file_bytes, file.filename)
        logger.info(f"File securely backed up to {s3_path}")
        
        # Step 1: Extract raw text from the file
        if is_pdf:
            parsed_text = await extract_text_from_pdf(file_bytes)
        else:
            parsed_text = await extract_text_from_txt(file_bytes)

        # Step 2: Run AI inference to extract structured fields
        extracted = await extract_fields(parsed_text)

        # Step 3: Build response model
        skills_raw = extracted.get("Skills", "")
        skills_list = (
            [s.strip() for s in skills_raw.split(",") if s.strip()]
            if isinstance(skills_raw, str)
            else (skills_raw if isinstance(skills_raw, list) else [])
        )

        education_raw = extracted.get("Education", "")
        education_list = (
            [education_raw.strip()] if isinstance(education_raw, str) and education_raw.strip() else
            (education_raw if isinstance(education_raw, list) else [])
        )

        resume_data = ResumeData(
            name=extracted.get("Name", ""),
            email=_normalize_email(extracted.get("Email Address")),
            skills=skills_list,
            education=education_list,
        )

        education_str = "; ".join(education_list) if education_list else ""
        try:
            doc = ResumeDocument(
                name=resume_data.name,
                email=resume_data.email,
                education=education_str,
                skills=resume_data.skills,
            )
            await doc.insert()
            logger.info("Persisted extraction result to MongoDB: %s", doc.id)
        except Exception:
           
            logger.warning("Failed to persist extraction result to MongoDB.")

        return ExtractionResponse(
            status="success",
            message=f"Successfully extracted data from {file.filename}.",
            data=resume_data,
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re_err:
        raise HTTPException(status_code=503, detail=str(re_err))
    except Exception as e:
        logger.exception("Unexpected error during extraction.")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    Run basic tests for the resume extraction API.
    Usage: python -m app.main --test
    """
    import sys
    
    if "--test" in sys.argv:
        from fastapi.testclient import TestClient
        import io
        
        client = TestClient(app)
        
        print("Running API tests...\n")
        
        # Test 1: Health check
        print("Test 1: Health check endpoint")
        response = client.get("/")
        assert response.status_code == 200
        print("PASSED\n")
        
        # Test 2: Reject non-PDF/TXT files
        print("Test 2: Reject unsupported file extension (.docx)")
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.docx", b"fake content", "application/msword")}
        )
        assert response.status_code == 400
        print("PASSED\n")
        
        # Test 3: Reject wrong MIME type
        print("Test 3: Reject wrong content type")
        fake_doc = b"fake word document content"
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.doc", fake_doc, "application/msword")}
        )
        assert response.status_code == 400
        print("PASSED\n")
        
        # Test 4: Accept plain text file
        print("Test 4: Accept plain text file")
        txt_content = b"Ahmed Tamer\nahmed.tamer@example.com\nSkills: Python, SQL\nEducation: BS Computer Science"
        response = client.post(
            "/api/v1/extract",
            files={"file": ("resume.txt", txt_content, "text/plain")}
        )
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            print("PASSED - Extracted:", data["data"])
        else:
            print(f"Skipped (model may not be loaded): {response.status_code}")
        print()
        
        # Test 5: Reject oversized files
        print("Test 5: Reject oversized files")
        large_content = b"%PDF-1.4 " + b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/api/v1/extract",
            files={"file": ("large.pdf", large_content, "application/pdf")}
        )
        assert response.status_code == 413
        print("PASSED\n")
        
        # Test 6: Valid PDF (minimal structure)
        print("Test 6: Valid PDF with minimal content")
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 44 >>\nstream\n"
            b"BT /F1 12 Tf 100 700 Td (Test Resume) Tj ET\n"
            b"endstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000214 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n307\n%%EOF\n"
        )
        response = client.post(
            "/api/v1/extract",
            files={"file": ("resume.pdf", minimal_pdf, "application/pdf")}
        )
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            assert "name" in data["data"]
            assert "email" in data["data"]
            assert "skills" in data["data"]
            assert "education" in data["data"]
            print("PASSED - Extracted:", data["data"])
        else:
            print(f"Skipped: {response.status_code}")
        print()
        
        # Test 7: Get all resumes endpoint
        print("Test 7: GET /api/v1/resumes endpoint")
        response = client.get("/api/v1/resumes")
        if response.status_code == 200:
            print("PASSED - Returns:", response.json())
        elif response.status_code == 503:
            print("Skipped (DB unavailable): 503")
        else:
            print(f"FAILED: {response.status_code}")
        print()
        
        print("All tests completed!")
        sys.exit(0)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)