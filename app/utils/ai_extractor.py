"""
AI-powered resume field extractor.

Loads a fine-tuned BERT + LoRA model for token classification (NER) and
extracts Name, Email Address, Skills, and Education from raw resume text.

Architecture (3-layer extraction stack):
    1. Fine-tuned BERT (LoRA) for structural fields:
         PERSON, EMAIL, DESIGNATION, EDUCATION, LOCATION
    2. Regex for contact info (overrides the model for EMAIL since
       pattern matching beats learning here).
    3. Dictionary lookup for SKILL against a curated taxonomy
       (resume-skill datasets are too noisy to learn from reliably).

The public surface that main.py depends on is:
    - load_model()
    - unload_model()
    - extract_fields(text) -> {Name, Email Address, Skills, Education}
"""

import os
import re
import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline as hf_pipeline, Pipeline
from app.utils.s3_helper import download_and_extract_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "bert-base-cased"
ADAPTER_PATH = "/app/final-resume-model"

# Long-document inference settings
MAX_INPUT_TOKENS = 384
STRIDE = 64

# Map model entity types -> API schema keys
# (DESIGNATION and LOCATION are extracted but not currently exposed in
# the response; left here so it's easy to add later without retraining.)
ENTITY_TO_FIELD = {
    "PERSON": "Name",
    "EMAIL": "Email Address",
    "EDUCATION": "Education",
}

# Regex for the contact-info layer
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DATE_RANGE_RE = re.compile(r"^\s*\d{4}\s*[-–—]\s*\d{4}\s*$")  # e.g. "2013-2017"
ALL_DIGITS_RE = re.compile(r"^[\d\s\-\–\—/().]+$")

# Skill dictionary (small, MVP-grade — swap in ESCO/O*NET later if needed).
# Lower-cased keys; multi-word skills are matched as whole-word phrases.
SKILL_TAXONOMY: List[str] = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "golang", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
    "matlab", "perl", "bash", "shell", "powershell",
    # Web / frontend
    "html", "css", "sass", "tailwind", "react", "next.js", "nextjs",
    "vue", "vue.js", "angular", "svelte", "jquery", "redux", "graphql",
    # Backend / frameworks
    "node.js", "nodejs", "express", "django", "flask", "fastapi", "spring",
    "spring boot", ".net", "asp.net", "ruby on rails", "laravel", "nestjs",
    # Data / ML
    "machine learning", "deep learning", "data science", "data analysis",
    "nlp", "natural language processing", "computer vision", "pytorch",
    "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
    "seaborn", "huggingface", "transformers", "lora", "rag", "llm",
    "opencv", "spacy", "xgboost", "lightgbm",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite",
    "oracle", "mariadb", "dynamodb", "cassandra", "elasticsearch",
    "snowflake", "bigquery",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "circleci", "ci/cd", "linux", "nginx", "apache", "kafka", "rabbitmq",
    "airflow", "spark", "hadoop",
    # Tools / general
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "figma",
    "rest api", "rest", "soap", "microservices", "agile", "scrum",
    "tdd", "unit testing", "selenium", "pytest", "junit",
]
# Pre-compile a single alternation regex (longest first so multi-word skills win)
_SKILL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+#.])(?:"
    + "|".join(sorted((re.escape(s) for s in SKILL_TAXONOMY), key=len, reverse=True))
    + r")(?![A-Za-z0-9+#])",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
class _ModelHolder:
    ner_pipeline: Optional[Pipeline] = None
    device: Optional[str] = None


_holder = _ModelHolder()


def load_model() -> None:
    if _holder.ner_pipeline is not None:
        logger.info("Model already loaded - skipping.")
        return

    adapter_path = os.path.abspath(ADAPTER_PATH)
    if not os.path.isdir(adapter_path) or not os.path.exists(os.path.join(adapter_path, "config.json")):
        logger.warning(f"Model not found at '{adapter_path}'. Fetching from S3...")
        download_and_extract_model(adapter_path)

    if torch.cuda.is_available():
        _holder.device = "cuda"
        device_arg = 0                          # pipeline takes GPU index, not string
        logger.info("CUDA available - using GPU: %s", torch.cuda.get_device_name(0))
    else:
        _holder.device = "cpu"
        device_arg = -1
        logger.info("No GPU found - running on CPU.")

    _holder.ner_pipeline = hf_pipeline(
        "token-classification",
        model=adapter_path,
        tokenizer=adapter_path,
        aggregation_strategy="simple",          # merges B-/I- subwords, gives entity_group
        device=device_arg,
    )
    logger.info("Resume extraction model ready on %s.", _holder.device)


def unload_model() -> None:
    _holder.ner_pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded.")


# ---------------------------------------------------------------------------
# Inference: NER with sliding window + offset-mapping alignment
# ---------------------------------------------------------------------------
def _predict_entities(text: str) -> List[Tuple[str, int, int, str]]:
    """
    Run the HF pipeline on `text` with stride-based long-document handling.
    Returns a list of (entity_type, start_char, end_char, span_text).

    `stride=STRIDE` makes the pipeline tokenize with overlapping windows
    (same as the manual sliding loop, but handled internally). Each result
    already carries character-level `start`/`end` so downstream code is
    unchanged.
    """
    raw: List[dict] = _holder.ner_pipeline(
        text,
        stride=STRIDE,                          # requires transformers >= 4.24
    )

    entities: List[Tuple[str, int, int, str]] = []
    for ent in raw:
        entity_type = ent["entity_group"]       # e.g. "PERSON", "EMAIL", "EDUCATION"
        start: int = ent["start"]
        end: int   = ent["end"]
        span: str  = text[start:end].strip()
        if span:
            entities.append((entity_type, start, end, span))

    return entities


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _is_garbage_person(span: str) -> bool:
    """Reject obvious non-person spans the model sometimes mislabels."""
    if not span or len(span) < 3:
        return True
    low = span.lower()
    if DATE_RANGE_RE.match(span) or ALL_DIGITS_RE.match(span):
        return True
    blacklist = ("university", "college", "institute", "school", "company",
                 "ltd", "inc", "llc", "gmbh", "corporation")
    return any(b in low for b in blacklist)


def _merge_contiguous(entities: List[Tuple[str, int, int, str]],
                      text: str) -> List[Tuple[str, int, int, str]]:
    """Merge same-type spans separated only by whitespace/punctuation."""
    if not entities:
        return entities
    entities = sorted(entities, key=lambda x: x[1])
    merged = [entities[0]]
    for typ, s, e, span in entities[1:]:
        ptyp, ps, pe, pspan = merged[-1]
        gap = text[pe:s]
        if typ == ptyp and gap.strip() == "" and len(gap) <= 3:
            merged[-1] = (typ, ps, e, text[ps:e].strip())
        else:
            merged.append((typ, s, e, span))
    return merged


def _dedupe(values: List[str]) -> List[str]:
    seen, out = set(), []
    for v in values:
        key = v.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out


def _extract_skills_from_taxonomy(text: str) -> List[str]:
    """Dictionary lookup for skills (the model doesn't predict SKILL)."""
    found: Dict[str, str] = {}                                   # lower -> original casing
    for m in _SKILL_PATTERN.finditer(text):
        canon = m.group(0).lower()
        if canon not in found:
            found[canon] = m.group(0)
    # Return canonical (taxonomy) casing for cleanliness
    canon_map = {s.lower(): s for s in SKILL_TAXONOMY}
    return [canon_map.get(k, v).title() if " " not in canon_map.get(k, v)
            else canon_map.get(k, v) for k, v in found.items()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def extract_fields(resume_text: str) -> Dict[str, str]:
    """
    Run the 3-layer extraction stack and return:
        {
            "Name":          "Ahmed Tamer",
            "Email Address": "ahmed@example.com",
            "Skills":        "Python, SQL, FastAPI",
            "Education":     "BS Computer Science; ..."
        }
    """
    if _holder.ner_pipeline is None:
        raise RuntimeError("Model is not loaded. Call load_model() at startup.")

    # Normalize away surrogate / weird Unicode that breaks tokenizers
    resume_text = resume_text.encode("utf-8", errors="ignore").decode("utf-8")

    def _run() -> Dict[str, str]:
        # Layer 1: NER
        raw = _predict_entities(resume_text)
        # print("Raw model output:", raw)
        raw = _merge_contiguous(raw, resume_text)

        buckets: Dict[str, List[str]] = {v: [] for v in ENTITY_TO_FIELD.values()}
        for typ, _s, _e, span in raw:
            field = ENTITY_TO_FIELD.get(typ)
            if not field:
                continue
            if field == "Name" and _is_garbage_person(span):
                continue
            buckets[field].append(span)

        # Dedupe
        for k in buckets:
            buckets[k] = _dedupe(buckets[k])

        # Layer 2: Email regex (overrides the model)
        regex_emails = _dedupe(EMAIL_RE.findall(resume_text))
        if regex_emails:
            buckets["Email Address"] = regex_emails

        # Layer 3: Skills via taxonomy lookup
        skills = _extract_skills_from_taxonomy(resume_text)

        result = {
            "Name": buckets["Name"][0] if buckets["Name"] else "",
            "Email Address": buckets["Email Address"][0] if buckets["Email Address"] else "",
            "Skills": ", ".join(skills),
            "Education": "; ".join(buckets["Education"]),
        }

        logger.debug(
            "Extracted: name=%r email=%r #skills=%d #edu=%d",
            result["Name"], result["Email Address"],
            len(skills), len(buckets["Education"]),
        )
        return result

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Smoke tests (run with: python -m app.utils.ai_extractor --test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        print("Running unit tests for post-processing...\n")

        # Garbage filter
        assert _is_garbage_person("2013-2017")
        assert _is_garbage_person("Stanford University")
        assert not _is_garbage_person("Ahmed Tamer")
        print("garbage filter: PASSED")

        # Skill taxonomy lookup
        text = "I work with Python, FastAPI and PostgreSQL daily. Also Machine Learning."
        skills = _extract_skills_from_taxonomy(text)
        assert "Python" in skills or "python" in [s.lower() for s in skills]
        assert any("postgres" in s.lower() for s in skills)
        assert any("machine learning" in s.lower() for s in skills)
        print("skill taxonomy: PASSED ->", skills)

        # Email regex
        emails = EMAIL_RE.findall("contact me at john.smith@gmail.com or x@y.co")
        assert "john.smith@gmail.com" in emails
        print("email regex: PASSED ->", emails)

        # Merge contiguous
        merged = _merge_contiguous(
            [("DESIGNATION", 0, 6, "Senior"),
             ("DESIGNATION", 7, 23, "Software Engineer")],
            "Senior Software Engineer",
        )
        assert merged == [("DESIGNATION", 0, 23, "Senior Software Engineer")]
        print("merge contiguous: PASSED")

        print("\nAll unit tests passed.")
        sys.exit(0)