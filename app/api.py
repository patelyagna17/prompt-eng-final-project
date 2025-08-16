# app/api.py
from typing import Any, Dict, List, Optional

import base64

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import HTTPException

from app.rag_pipeline import RagService
from app.pinecone_embeds import query_pinecone, upsert_texts
from app.utils import SETTINGS


app = FastAPI(
    title="GenAI Doc Assistant API",
    version="0.1.3",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (open for local dev; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single service instance
service = RagService()
service.ensure_kb()


# -------------------- Models --------------------
class AskReq(BaseModel):
    question: str = Field(..., min_length=3, description="Your question")
    k: int = Field(6, ge=1, le=20, description="Top-k documents to retrieve")


class AskPineconeReq(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=50)
    namespace: Optional[str] = None


# app/api.py  (near the top with your Pydantic models)
class SummarizeReq(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text extracted from a file")
    max_words: int = Field(250, ge=50, le=1000, description="Summary length budget")


class UpsertPineconeReq(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    ids: List[str] = Field(..., min_items=1)
    namespace: Optional[str] = Field(None, description="Optional Pinecone namespace")


# -------------------- Routes --------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "GenAI Doc Assistant API. See /docs for Swagger UI."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> Dict[str, Any]:
    """Quick sanity check that your server picked up .env."""
    return {
        "llm_backend": SETTINGS.llm_backend,
        "embeddings_backend": SETTINGS.embeddings_backend,
        "has_key": bool(SETTINGS.openai_api_key),
    }


@app.post("/ingest")
def ingest() -> Dict[str, Any]:
    """(Re)build the vector store from data/raw/."""
    try:
        return service.ingest()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask")
def ask(req: AskReq) -> Dict[str, Any]:
    q = req.question.strip()
    if not q:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be empty.",
        )
    try:
        return service.answer(q, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

# app/api.py
@app.post("/summarize")
def summarize(req: SummarizeReq) -> Dict[str, Any]:
    """Summarize arbitrary text (used by the Streamlit upload tab)."""
    try:
        txt = (req.text or "").strip()
        if len(txt) < 20:
            # Don’t 422; return something useful for small snippets
            return {
                "summary": txt if txt else "(Document is empty.)",
                "bullets": [],
                "table_csv": None,
            }
        return service.summarize_text(txt, max_words=req.max_words)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/ask_pinecone")
def ask_pinecone(req: AskPineconeReq) -> Dict[str, Any]:
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    try:
        res = query_pinecone(q, top_k=req.top_k, namespace=req.namespace)
        matches = res.get("matches", [])
        chunks: list[tuple[str, str]] = []
        for m in matches:
            meta = m.get("metadata") or {}
            text = meta.get("text", "")
            src = meta.get("source") or f"pinecone:{m.get('id', '')}"
            if text:
                chunks.append((text, src))
        if not chunks:
            return {"answer": "No relevant results in Pinecone.", "sources": []}
        return service.answer_from_texts(q, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/upsert_pinecone")
def upsert_pinecone(req: UpsertPineconeReq) -> JSONResponse:
    if not req.texts or not req.ids:
        raise HTTPException(status_code=422, detail="texts and ids must be non-empty")
    if len(req.texts) != len(req.ids):
        raise HTTPException(status_code=422, detail="texts and ids must be same length")

    try:
        res = upsert_texts(req.texts, req.ids, namespace=req.namespace)

        # Robustly make it JSON-safe
        def to_safe(obj):
            try:
                return jsonable_encoder(obj)
            except Exception:
                try:
                    if hasattr(obj, "to_dict"):
                        return obj.to_dict()
                    if hasattr(obj, "dict"):
                        return obj.dict()
                    if hasattr(obj, "model_dump"):
                        return obj.model_dump()
                except Exception:
                    pass
                # Last resort: human-readable string
                return repr(obj)

        safe = to_safe(res) if res is not None else {"ok": True}
        return JSONResponse({"status": "ok", "upserted": len(req.ids), "result": safe})

    except Exception as e:
        # Always return clean JSON on failure
        return JSONResponse(status_code=500, content={"detail": str(e)})

# -------------------- Image description (photo summary) --------------------
@app.post("/describe_image")
async def describe_image(
    prompt: str = Form("Describe the image. If there are charts or tables, summarize the key numbers accurately."),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Accepts an image via multipart form-data and returns a concise description
    using an OpenAI vision model (if API key is present).
    """
    try:
        content_type = file.content_type or "image/png"
        img_bytes = await file.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        if not SETTINGS.openai_api_key:
            return {
                "answer": "(OpenAI key missing) Received your image but cannot call a vision model. "
                          "Set OPENAI_API_KEY in your environment."
            }

        # Call OpenAI vision-capable model
        from openai import OpenAI  # import here to avoid hard dependency if key missing
        client = OpenAI()

        resp = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{b64}"},
                },
            ],
        }
    ],
)
        answer = (resp.choices[0].message.content or "").strip()
        return {"answer": answer or "(no answer)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e