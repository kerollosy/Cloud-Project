import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from app.db_models import ResumeDocument

logger = logging.getLogger(__name__)


async def init_db():
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DATABASE_NAME")

    if not mongo_url or not db_name:
        logger.warning("Missing MongoDB configuration - DB persistence disabled")
        return

    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]

        await init_beanie(
            database=db,
            document_models=[ResumeDocument],
        )
        logger.info("Connected to MongoDB: %s", db_name)

    except Exception as e:
        logger.error("Failed to connect to MongoDB: %s", str(e))
