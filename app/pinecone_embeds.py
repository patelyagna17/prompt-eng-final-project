# # app/pinecone_embeds.py
# import os
# from typing import List, Dict, Any
# from pinecone import Pinecone, ServerlessSpec
# from langchain_openai import OpenAIEmbeddings

# INDEX_NAME = os.getenv("PINECONE_INDEX", "genai-doc-assistant")
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# # Use the same embedding model as the rest of the app
# _EMB = OpenAIEmbeddings(model="text-embedding-3-large")

# def ensure_index(dimension: int = 3072, metric: str = "cosine"):
#     names = [i.name for i in pc.list_indexes()]
#     if INDEX_NAME not in names:
#         pc.create_index(
#             name=INDEX_NAME,
#             dimension=dimension,
#             metric=metric,
#             spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
#                                 region=os.getenv("PINECONE_REGION", "us-east-1")),
#         )

# def upsert_texts(texts: List[str], ids: List[str]) -> Dict[str, Any]:
#     """Upsert plain texts with metadata {'text': ...}."""
#     ensure_index()
#     index = pc.Index(INDEX_NAME)
#     vectors = _EMB.embed_documents(texts)
#     payload = [
#         {"id": _id, "values": vec, "metadata": {"text": txt}}
#         for _id, vec, txt in zip(ids, vectors, texts)
#     ]
#     return index.upsert(vectors=payload)

# def query_pinecone(query_text: str, top_k: int = 10) -> Dict[str, Any]:
#     """Query Pinecone and return raw results (with metadata)."""
#     ensure_index()
#     index = pc.Index(INDEX_NAME)
#     vec = _EMB.embed_query(query_text)
#     return index.query(vector=vec, top_k=top_k, include_metadata=True)
# app/pinecone_embeds.py
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

INDEX_NAME = os.getenv("PINECONE_INDEX", "genai-doc-assistant")
DEFAULT_NS = os.getenv("PINECONE_NAMESPACE", "")  # optional
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Use the same embedding model as the rest of the app
_EMB = OpenAIEmbeddings(model="text-embedding-3-large")

def ensure_index(dimension: int = 3072, metric: str = "cosine"):
    names = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )

def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    parts, i, n = [], 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        parts.append(text[i:i+size])
        i += step
    return parts

def upsert_texts(
    texts: List[str],
    ids: List[str],
    namespace: Optional[str] = None,
    metas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Upsert plain texts with metadata. `ids` must match length of `texts`."""
    ensure_index()
    index = pc.Index(INDEX_NAME)
    vectors = _EMB.embed_documents(texts)

    payload = []
    for idx, (tid, vec, txt) in enumerate(zip(ids, vectors, texts)):
        meta = {"text": txt, "source": tid, "chunk": idx, "type": "text"}
        if metas and idx < len(metas) and isinstance(metas[idx], dict):
            meta.update(metas[idx])
        payload.append({"id": tid, "values": vec, "metadata": meta})

    return index.upsert(vectors=payload, namespace=namespace or DEFAULT_NS)

def query_pinecone(
    query_text: str,
    top_k: int = 10,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """Query Pinecone and return raw results (with metadata)."""
    ensure_index()
    index = pc.Index(INDEX_NAME)
    vec = _EMB.embed_query(query_text)
    return index.query(
        vector=vec, top_k=top_k, include_metadata=True, namespace=namespace or DEFAULT_NS
    )
