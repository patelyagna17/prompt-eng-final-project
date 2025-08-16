# app/utils.py
import os
import time
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("genai-doc-assistant")


@dataclass
class Settings:
    embeddings_backend: str = os.getenv("EMBEDDINGS_BACKEND", "local")  # local | openai
    llm_backend: str = os.getenv("LLM_BACKEND", "local")  # local | openai
    vector_db: str = os.getenv(
        "VECTOR_DB", "faiss"
    )  # faiss | chroma (faiss used below)
    raw_dir: str = os.getenv("RAW_DIR", "./data/raw")
    processed_dir: str = os.getenv("PROCESSED_DIR", "./data/processed")
    kb_dir: str = os.getenv("KB_DIR", "./data/kb")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


SETTINGS = Settings()


def timed(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = (time.time() - t0) * 1000
        log.info(f"{fn.__name__} took {dt:.1f} ms")
        return out

    return wrapper
