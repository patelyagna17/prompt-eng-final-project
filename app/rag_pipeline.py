# # # # app/rag_pipeline.py
# # # import os
# # # from pathlib import Path
# # # from typing import List, Tuple, Dict, Any

# # # from app.utils import SETTINGS, log, timed
# # # from app.loaders import load_docs
# # # from app.chunkers import split_docs
# # # from app.prompts import SYSTEM_INSTRUCTIONS, ANSWER_PROMPT

# # # from langchain.docstore.document import Document
# # # from langchain_community.vectorstores import FAISS
# # # from langchain_community.embeddings import HuggingFaceEmbeddings
# # # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# # # from langchain.schema import SystemMessage, HumanMessage

# # # # --- Embeddings factory -------------------------------------------------------

# # # def _embeddings():
# # #     if SETTINGS.embeddings_backend.lower() == "openai" and SETTINGS.openai_api_key:
# # #         return OpenAIEmbeddings(model="text-embedding-3-large")
# # #     # local default: fast & CPU-friendly
# # #     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # # # --- LLM factory --------------------------------------------------------------

# # # class _LocalTextGen:
# # #     """Tiny wrapper for local text2text models via transformers."""
# # #     def __init__(self, model_name: str = "google/flan-t5-base", device: str = None):
# # #         from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# # #         tok = AutoTokenizer.from_pretrained(model_name)
# # #         mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# # #         self.pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)

# # #     def invoke(self, prompt: str) -> str:
# # #         out = self.pipe(prompt, max_new_tokens=512)[0]["generated_text"]
# # #         return out

# # # def _llm():
# # #     if SETTINGS.llm_backend.lower() == "openai" and SETTINGS.openai_api_key:
# # #         return ChatOpenAI(model="gpt-4o-mini", temperature=0)
# # #     return _LocalTextGen()  # dev-friendly local fallback

# # # # --- RAG service --------------------------------------------------------------

# # # class RagService:
# # #     def __init__(self):
# # #         self.emb = _embeddings()
# # #         self.llm = _llm()
# # #         self.vs = None  # FAISS store
# # #         Path(SETTINGS.kb_dir).mkdir(parents=True, exist_ok=True)

# # #     # ---------- KB building / loading ----------
# # #     @timed
# # #     def ingest(self) -> Dict[str, Any]:
# # #         docs = load_docs(SETTINGS.raw_dir)
# # #         chunks = split_docs(docs)
# # #         if not chunks:
# # #             return {"docs": 0, "chunks": 0, "kb_dir": SETTINGS.kb_dir, "note": "No input files found in data/raw"}
# # #         self.vs = FAISS.from_documents(chunks, self.emb)
# # #         self.vs.save_local(SETTINGS.kb_dir)
# # #         return {"docs": len(docs), "chunks": len(chunks), "kb_dir": SETTINGS.kb_dir}

# # #     def _faiss_exists(self) -> bool:
# # #         # FAISS saves index + index.pkl; existence of folder is enough to try load
# # #         return Path(SETTINGS.kb_dir).exists() and any(Path(SETTINGS.kb_dir).glob("*"))

# # #     @timed
# # #     def load_or_raise(self):
# # #         if not self._faiss_exists():
# # #             raise RuntimeError("KB not built yet. Run ingest first.")
# # #         self.vs = FAISS.load_local(
# # #             SETTINGS.kb_dir, self.emb, allow_dangerous_deserialization=True
# # #         )

# # #     def ensure_kb(self):
# # #         if self.vs is None:
# # #             if self._faiss_exists():
# # #                 self.load_or_raise()
# # #             else:
# # #                 log.info("KB missing; ingesting from data/raw ...")
# # #                 self.ingest()

# # #     # ---------- Retrieval + Generation ----------
# # #     def _retrieve(self, question: str, k: int = 6) -> List[Document]:
# # #         self.ensure_kb()
# # #         retriever = self.vs.as_retriever(search_kwargs={"k": k})
# # #         return retriever.get_relevant_documents(question)

# # #     @staticmethod
# # #     def _format_context(docs: List[Document], max_chars: int = 1200) -> Tuple[str, List[str]]:
# # #         pieces, sources = [], []
# # #         for d in docs:
# # #             src = str(d.metadata.get("source", "unknown"))
# # #             sources.append(src)
# # #             # keep context compact to leave room for generation
# # #             text = d.page_content.strip().replace("\n\n", "\n")
# # #             pieces.append(f"### Source: {src}\n{text[:max_chars]}")
# # #         return "\n\n".join(pieces), list(dict.fromkeys(sources))  # dedupe order

# # #     @timed
# # #     def answer(self, question: str, k: int = 6) -> Dict[str, Any]:
# # #         docs = self._retrieve(question, k=k)
# # #         context, sources = self._format_context(docs)
# # #         prompt_text = f"{SYSTEM_INSTRUCTIONS}\n\n" + ANSWER_PROMPT.format(question=question, context=context)

# # #         # OpenAI chat vs local text2text
# # #         if isinstance(self.llm, ChatOpenAI):
# # #             messages = [SystemMessage(content=SYSTEM_INSTRUCTIONS),
# # #                         HumanMessage(content=ANSWER_PROMPT.format(question=question, context=context))]
# # #             out = self.llm.invoke(messages).content
# # #         else:
# # #             out = self.llm.invoke(prompt_text)

# # #         return {"answer": out, "sources": sources}
# # # app/rag_pipeline.py
# # from pathlib import Path
# # from typing import List, Tuple, Dict, Any

# # from app.utils import SETTINGS, log, timed
# # from app.loaders import load_docs
# # from app.chunkers import split_docs
# # from app.prompts import SYSTEM_INSTRUCTIONS, ANSWER_PROMPT

# # from langchain.docstore.document import Document
# # from langchain_community.vectorstores import FAISS
# # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# # from langchain.schema import SystemMessage, HumanMessage
# # from langchain.retrievers import ContextualCompressionRetriever
# # from langchain.retrievers.document_compressors import LLMChainExtractor
# # from langchain_huggingface import HuggingFaceEmbeddings


# # # --- Embeddings factory -------------------------------------------------------
# # def _embeddings():
# #     if SETTINGS.embeddings_backend.lower() == "openai" and SETTINGS.openai_api_key:
# #         return OpenAIEmbeddings(model="text-embedding-3-large")
# #     # local default: fast & CPU-friendly
# #     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# # # --- LLM factory --------------------------------------------------------------
# # class _LocalTextGen:
# #     """Tiny wrapper for local text2text models via transformers."""

# #     def __init__(self, model_name: str = "google/flan-t5-base", device: str = None):
# #         from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# #         tok = AutoTokenizer.from_pretrained(model_name)
# #         mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# #         self.pipe = pipeline(
# #             "text2text-generation", model=mdl, tokenizer=tok, device=device
# #         )

# #     def invoke(self, prompt: str) -> str:
# #         out = self.pipe(prompt, max_new_tokens=512)[0]["generated_text"]
# #         return out


# # def _llm():
# #     if SETTINGS.llm_backend.lower() == "openai" and SETTINGS.openai_api_key:
# #         return ChatOpenAI(model="gpt-4o-mini", temperature=0)
# #     return _LocalTextGen()  # dev-friendly local fallback


# # # --- RAG service --------------------------------------------------------------
# # class RagService:
# #     def __init__(self):
# #         self.emb = _embeddings()
# #         self.llm = _llm()
# #         self.vs = None  # FAISS store
# #         Path(SETTINGS.kb_dir).mkdir(parents=True, exist_ok=True)

# #     # ---------- KB building / loading ----------
# #     @timed
# #     def ingest(self) -> Dict[str, Any]:
# #         docs = load_docs(SETTINGS.raw_dir)
# #         chunks = split_docs(docs)
# #         if not chunks:
# #             return {
# #                 "docs": 0,
# #                 "chunks": 0,
# #                 "kb_dir": SETTINGS.kb_dir,
# #                 "note": "No input files found in data/raw",
# #             }
# #         self.vs = FAISS.from_documents(chunks, self.emb)
# #         self.vs.save_local(SETTINGS.kb_dir)
# #         return {"docs": len(docs), "chunks": len(chunks), "kb_dir": SETTINGS.kb_dir}

# #     def _faiss_exists(self) -> bool:
# #         # FAISS saves index + index.pkl; any file in folder -> try load
# #         return Path(SETTINGS.kb_dir).exists() and any(Path(SETTINGS.kb_dir).glob("*"))

# #     @timed
# #     def load_or_raise(self):
# #         if not self._faiss_exists():
# #             raise RuntimeError("KB not built yet. Run ingest first.")
# #         self.vs = FAISS.load_local(
# #             SETTINGS.kb_dir, self.emb, allow_dangerous_deserialization=True
# #         )

# #     def ensure_kb(self):
# #         if self.vs is None:
# #             if self._faiss_exists():
# #                 self.load_or_raise()
# #             else:
# #                 log.info("KB missing; ingesting from data/raw ...")
# #                 self.ingest()

# #     # ---------- Retrieval + Generation ----------
# #     def _retrieve(self, question: str, k: int = 6) -> List[Document]:
# #         self.ensure_kb()
# #         base = self.vs.as_retriever(
# #             search_type="mmr", search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
# #         )
# #         compressor = LLMChainExtractor.from_llm(
# #             ChatOpenAI(model="gpt-4o-mini", temperature=0)
# #         )
# #         retriever = ContextualCompressionRetriever(
# #             base_retriever=base, base_compressor=compressor
# #         )
# #         return retriever.invoke(question)

# #     @staticmethod
# #     def _format_context(
# #         docs: List[Document], max_chars: int = 1200
# #     ) -> Tuple[str, List[str]]:
# #         pieces, sources = [], []
# #         for d in docs:
# #             src = str(d.metadata.get("source", "unknown"))
# #             sources.append(src)
# #             # keep context compact to leave room for generation
# #             text = d.page_content.strip().replace("\n\n", "\n")
# #             pieces.append(f"### Source: {src}\n{text[:max_chars]}")
# #         return "\n\n".join(pieces), list(dict.fromkeys(sources))  # dedupe order

# #     @timed
# #     def answer(self, question: str, k: int = 6) -> Dict[str, Any]:
# #         docs = self._retrieve(question, k=k)
# #         context, sources = self._format_context(docs)
# #         prompt_text = f"{SYSTEM_INSTRUCTIONS}\n\n" + ANSWER_PROMPT.format(
# #             question=question, context=context
# #         )

# #         # OpenAI chat vs local text2text
# #         if isinstance(self.llm, ChatOpenAI):
# #             messages = [
# #                 SystemMessage(content=SYSTEM_INSTRUCTIONS),
# #                 HumanMessage(
# #                     content=ANSWER_PROMPT.format(question=question, context=context)
# #                 ),
# #             ]
# #             out = self.llm.invoke(messages).content
# #         else:
# #             out = self.llm.invoke(prompt_text)

# #         # Append sources for readability
# #         text_sources = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
# #         return {"answer": out + text_sources, "sources": sources}
# # app/rag_pipeline.py
# from pathlib import Path
# from typing import List, Tuple, Dict, Any
# import json  # for parsing summarizer JSON

# from app.utils import SETTINGS, log, timed
# from app.loaders import load_docs
# from app.chunkers import split_docs
# from app.prompts import SYSTEM_INSTRUCTIONS, ANSWER_PROMPT

# from langchain.docstore.document import Document
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain_huggingface import HuggingFaceEmbeddings


# # --- Embeddings factory -------------------------------------------------------
# def _embeddings():
#     if SETTINGS.embeddings_backend.lower() == "openai" and SETTINGS.openai_api_key:
#         return OpenAIEmbeddings(model="text-embedding-3-large")
#     # local default: fast & CPU-friendly
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# # --- LLM factory --------------------------------------------------------------
# class _LocalTextGen:
#     """Tiny wrapper for local text2text models via transformers."""

#     def __init__(self, model_name: str = "google/flan-t5-base", device: str = None):
#         from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

#         tok = AutoTokenizer.from_pretrained(model_name)
#         mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         self.pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)

#     def invoke(self, prompt: str) -> str:
#         out = self.pipe(prompt, max_new_tokens=512)[0]["generated_text"]
#         return out


# def _llm():
#     if SETTINGS.llm_backend.lower() == "openai" and SETTINGS.openai_api_key:
#         return ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     return _LocalTextGen()  # dev-friendly local fallback


# # --- RAG service --------------------------------------------------------------
# class RagService:
#     def __init__(self):
#         self.emb = _embeddings()
#         self.llm = _llm()
#         self.vs = None  # FAISS store
#         Path(SETTINGS.kb_dir).mkdir(parents=True, exist_ok=True)

#     # ---------- KB building / loading ----------
#     @timed
#     def ingest(self) -> Dict[str, Any]:
#         docs = load_docs(SETTINGS.raw_dir)
#         chunks = split_docs(docs)
#         if not chunks:
#             return {
#                 "docs": 0,
#                 "chunks": 0,
#                 "kb_dir": SETTINGS.kb_dir,
#                 "note": "No input files found in data/raw",
#             }
#         self.vs = FAISS.from_documents(chunks, self.emb)
#         self.vs.save_local(SETTINGS.kb_dir)
#         return {"docs": len(docs), "chunks": len(chunks), "kb_dir": SETTINGS.kb_dir}

#     def _faiss_exists(self) -> bool:
#         # FAISS saves index + index.pkl; any file in folder -> try load
#         return Path(SETTINGS.kb_dir).exists() and any(Path(SETTINGS.kb_dir).glob("*"))

#     @timed
#     def load_or_raise(self):
#         if not self._faiss_exists():
#             raise RuntimeError("KB not built yet. Run ingest first.")
#         self.vs = FAISS.load_local(
#             SETTINGS.kb_dir, self.emb, allow_dangerous_deserialization=True
#         )

#     def ensure_kb(self):
#         if self.vs is None:
#             if self._faiss_exists():
#                 self.load_or_raise()
#             else:
#                 log.info("KB missing; ingesting from data/raw ...")
#                 self.ingest()

#     # ---------- Retrieval + Generation ----------
#     def _retrieve(self, question: str, k: int = 6) -> List[Document]:
#         """MMR retrieval with optional LLM compression (only when using ChatOpenAI)."""
#         self.ensure_kb()

#         base = self.vs.as_retriever(
#             search_type="mmr",
#             search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
#         )

#         # Only use compression if our active LLM is ChatOpenAI (we have a key).
#         if isinstance(self.llm, ChatOpenAI):
#             compressor = LLMChainExtractor.from_llm(self.llm)
#             retriever = ContextualCompressionRetriever(
#                 base_retriever=base, base_compressor=compressor
#             )
#             return retriever.invoke(question)

#         # local fallback: no compression
#         return base.invoke(question)

#     @staticmethod
#     def _format_context(docs: List[Document], max_chars: int = 1200) -> Tuple[str, List[str]]:
#         pieces, sources = [], []
#         for d in docs:
#             src = str(d.metadata.get("source", "unknown"))
#             sources.append(src)
#             # keep context compact to leave room for generation
#             text = d.page_content.strip().replace("\n\n", "\n")
#             pieces.append(f"### Source: {src}\n{text[:max_chars]}")
#         return "\n\n".join(pieces), list(dict.fromkeys(sources))  # dedupe order

#     @timed
#     def answer_from_texts(self, question: str, chunks: list[tuple[str, str]]) -> Dict[str, Any]:
#         """Generate an answer using provided (text, source) chunks; skips vector retrieval."""
#         # Build temporary docs from (text, source) pairs
#         docs = [Document(page_content=t, metadata={"source": s}) for t, s in chunks if t]
#         if not docs:
#             return {"answer": "No context provided.", "sources": []}

#         # Reuse context formatter
#         context, sources = self._format_context(docs)

#         # OpenAI chat vs local text2text
#         if isinstance(self.llm, ChatOpenAI):
#             messages = [
#                 SystemMessage(content=SYSTEM_INSTRUCTIONS),
#                 HumanMessage(content=ANSWER_PROMPT.format(question=question, context=context)),
#             ]
#             out = self.llm.invoke(messages).content
#         else:
#             prompt_text = f"{SYSTEM_INSTRUCTIONS}\n\n" + ANSWER_PROMPT.format(
#                 question=question, context=context
#             )
#             out = self.llm.invoke(prompt_text)

#         # Match your existing style: include sources text + structured sources
#         text_sources = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
#         return {"answer": out + text_sources, "sources": sources}


#     # ---------- Helpers for long text ----------
#     def _chunk_text(self, text: str, max_len: int = 3500, overlap: int = 300) -> list[str]:
#         out, i, n = [], 0, len(text)
#         while i < n:
#             out.append(text[i:i + max_len])
#             i += max_len - overlap
#         return out

#     # ---------- Summarization (used by /summarize) ----------
#     def summarize_text(self, text: str, max_words: int = 250) -> Dict[str, Any]:
#         """
#         Summarize arbitrary text and, when possible, extract a tiny CSV table.
#         Returns: {"summary": str, "bullets": [..], "table_csv": str, "raw": str}
#         """
#         import re  # local import ok; or move to top
#         SYS = (
#             "You are a precise analyst. Summarize the following content.\n"
#             f"- Keep the prose summary <= {max_words} words.\n"
#             "- Add 3–7 concise bullet points.\n"
#             "- If you see a small numeric table, output it as CSV (<=6 rows, <=6 cols), "
#             "else set table_csv to empty string.\n"
#             "- Return STRICT JSON with keys: summary (string), bullets (list of strings), table_csv (string)."
#         )

#         # If very long content and ChatOpenAI is available, do chunked mini-summaries then combine.
#         if len(text) > 14000 and isinstance(self.llm, ChatOpenAI):
#             parts = self._chunk_text(text, max_len=3500, overlap=300)
#             mini_summaries, bullets_all, tbls = [], [], []
#             for p in parts:
#                 res = self.summarize_text(p, max_words=max(120, max_words // max(len(parts), 1)))
#                 mini_summaries.append(res.get("summary", ""))
#                 bullets_all.extend(res.get("bullets", [])[:3])
#                 if res.get("table_csv"):
#                     tbls.append(res["table_csv"])
#             combined = "\n\n".join(mini_summaries)
#             table_csv = tbls[0] if tbls else ""
#             final = self.summarize_text(combined, max_words=max_words)
#             # prefer first detected table from parts
#             if table_csv and not final.get("table_csv"):
#                 final["table_csv"] = table_csv
#             return final

#         user = text[:16000]  # safety on context length

#         # Use JSON mode ONLY here when using ChatOpenAI
#         if isinstance(self.llm, ChatOpenAI):
#             llm_json = self.llm.bind(response_format={"type": "json_object"})
#             out = llm_json.invoke([SystemMessage(content=SYS), HumanMessage(content=user)]).content
#         else:
#             out = self.llm.invoke(SYS + "\n\nCONTENT:\n" + user)

#         # Robust parse: try JSON -> try to extract trailing JSON block -> fallback to prose
#         payload = {"summary": out.strip(), "bullets": [], "table_csv": "", "raw": out}
#         try:
#             data = json.loads(out)
#         except Exception:
#             m = re.search(r"\{[\s\S]*\}$", out)  # last JSON-looking block
#             data = json.loads(m.group(0)) if m else {}
#         if isinstance(data, dict):
#             payload.update({
#                 "summary": data.get("summary", payload["summary"]),
#                 "bullets": data.get("bullets", []),
#                 "table_csv": data.get("table_csv", ""),
#             })
#         return payload

# app/rag_pipeline.py
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json  # for parsing summarizer JSON

from app.utils import SETTINGS, log, timed
from app.loaders import load_docs
from app.chunkers import split_docs
from app.prompts import SYSTEM_INSTRUCTIONS, ANSWER_PROMPT

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_huggingface import HuggingFaceEmbeddings


# --- Embeddings factory -------------------------------------------------------
def _embeddings():
    """Return the embedding function based on settings."""
    if SETTINGS.embeddings_backend.lower() == "openai" and SETTINGS.openai_api_key:
        # 3072-dim; matches your Pinecone config
        return OpenAIEmbeddings(model="text-embedding-3-large")
    # local default: fast & CPU-friendly
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- LLM factory --------------------------------------------------------------
class _LocalTextGen:
    """Tiny wrapper for local text2text models via transformers."""

    def __init__(self, model_name: str = "google/flan-t5-base", device: str | None = None):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)

    def invoke(self, prompt: str) -> str:
        out = self.pipe(prompt, max_new_tokens=512)[0]["generated_text"]
        return out


def _llm():
    """Return the generation model based on settings."""
    if SETTINGS.llm_backend.lower() == "openai" and SETTINGS.openai_api_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _LocalTextGen()  # dev-friendly local fallback


# --- RAG service --------------------------------------------------------------
class RagService:
    def __init__(self):
        self.emb = _embeddings()
        self.llm = _llm()
        self.vs = None  # FAISS store
        Path(SETTINGS.kb_dir).mkdir(parents=True, exist_ok=True)

    # ---------- KB building / loading ----------
    @timed
    def ingest(self) -> Dict[str, Any]:
        docs = load_docs(SETTINGS.raw_dir)
        chunks = split_docs(docs)
        if not chunks:
            return {
                "docs": 0,
                "chunks": 0,
                "kb_dir": SETTINGS.kb_dir,
                "note": "No input files found in data/raw",
            }
        self.vs = FAISS.from_documents(chunks, self.emb)
        self.vs.save_local(SETTINGS.kb_dir)
        return {"docs": len(docs), "chunks": len(chunks), "kb_dir": SETTINGS.kb_dir}

    def _faiss_exists(self) -> bool:
        # FAISS saves index + index.pkl; any file in folder -> try load
        return Path(SETTINGS.kb_dir).exists() and any(Path(SETTINGS.kb_dir).glob("*"))

    @timed
    def load_or_raise(self):
        if not self._faiss_exists():
            raise RuntimeError("KB not built yet. Run ingest first.")
        self.vs = FAISS.load_local(
            SETTINGS.kb_dir, self.emb, allow_dangerous_deserialization=True
        )

    def ensure_kb(self):
        if self.vs is None:
            if self._faiss_exists():
                self.load_or_raise()
            else:
                log.info("KB missing; ingesting from data/raw ...")
                self.ingest()

    # ---------- Retrieval + Generation ----------
    def _retrieve(self, question: str, k: int = 6) -> List[Document]:
        """MMR retrieval with optional LLM compression (only when using ChatOpenAI)."""
        self.ensure_kb()

        base = self.vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
        )

        # Only use compression if our active LLM is ChatOpenAI (we have a key).
        if isinstance(self.llm, ChatOpenAI):
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_retriever=base, base_compressor=compressor
            )
            return retriever.invoke(question)

        # local fallback: no compression
        return base.invoke(question)

    @staticmethod
    def _format_context(docs: List[Document], max_chars: int = 1200) -> Tuple[str, List[str]]:
        pieces, sources = [], []
        for d in docs:
            src = str(d.metadata.get("source", "unknown"))
            sources.append(src)
            # keep context compact to leave room for generation
            text = d.page_content.strip().replace("\n\n", "\n")
            pieces.append(f"### Source: {src}\n{text[:max_chars]}")
        # dedupe sources preserving order
        return "\n\n".join(pieces), list(dict.fromkeys(sources))

    @timed
    def answer(self, question: str, k: int = 6) -> Dict[str, Any]:
        """Retrieve from FAISS and generate an answer with sources."""
        # 1) retrieve docs
        docs = self._retrieve(question, k=k)

        # 2) format context for the LLM
        context, sources = self._format_context(docs)

        # 3) generate with either OpenAI Chat or local fallback
        if isinstance(self.llm, ChatOpenAI):
            messages = [
                SystemMessage(content=SYSTEM_INSTRUCTIONS),
                HumanMessage(content=ANSWER_PROMPT.format(question=question, context=context)),
            ]
            out = self.llm.invoke(messages).content
        else:
            prompt_text = f"{SYSTEM_INSTRUCTIONS}\n\n" + ANSWER_PROMPT.format(
                question=question, context=context
            )
            out = self.llm.invoke(prompt_text)

        # 4) include sources in the answer text and also separately
        text_sources = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
        return {"answer": out + text_sources, "sources": sources}

    @timed
    def answer_from_texts(self, question: str, chunks: list[tuple[str, str]]) -> Dict[str, Any]:
        """Generate an answer using provided (text, source) chunks; skips vector retrieval."""
        # Build temporary docs from (text, source) pairs
        docs = [Document(page_content=t, metadata={"source": s}) for t, s in chunks if t]
        if not docs:
            return {"answer": "No context provided.", "sources": []}

        # Reuse context formatter
        context, sources = self._format_context(docs)

        # OpenAI chat vs local text2text
        if isinstance(self.llm, ChatOpenAI):
            messages = [
                SystemMessage(content=SYSTEM_INSTRUCTIONS),
                HumanMessage(content=ANSWER_PROMPT.format(question=question, context=context)),
            ]
            out = self.llm.invoke(messages).content
        else:
            prompt_text = f"{SYSTEM_INSTRUCTIONS}\n\n" + ANSWER_PROMPT.format(
                question=question, context=context
            )
            out = self.llm.invoke(prompt_text)

        # Match your existing style: include sources text + structured sources
        text_sources = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
        return {"answer": out + text_sources, "sources": sources}

    # ---------- Helpers for long text ----------
    def _chunk_text(self, text: str, max_len: int = 3500, overlap: int = 300) -> list[str]:
        out, i, n = [], 0, len(text)
        while i < n:
            out.append(text[i : i + max_len])
            i += max_len - overlap
        return out

    # ---------- Summarization (used by /summarize) ----------
    def summarize_text(self, text: str, max_words: int = 250) -> Dict[str, Any]:
        """
        Summarize arbitrary text and, when possible, extract a tiny CSV table.
        Returns: {"summary": str, "bullets": [..], "table_csv": str, "raw": str}
        """
        import re  # local import ok; or move to top
        SYS = (
            "You are a precise analyst. Summarize the following content.\n"
            f"- Keep the prose summary <= {max_words} words.\n"
            "- Add 3–7 concise bullet points.\n"
            "- If you see a small numeric table, output it as CSV (<=6 rows, <=6 cols), "
            "else set table_csv to empty string.\n"
            "- Return STRICT JSON with keys: summary (string), bullets (list of strings), table_csv (string)."
        )

        # If very long content and ChatOpenAI is available, do chunked mini-summaries then combine.
        if len(text) > 14000 and isinstance(self.llm, ChatOpenAI):
            parts = self._chunk_text(text, max_len=3500, overlap=300)
            mini_summaries, bullets_all, tbls = [], [], []
            for p in parts:
                res = self.summarize_text(p, max_words=max(120, max_words // max(len(parts), 1)))
                mini_summaries.append(res.get("summary", ""))
                bullets_all.extend(res.get("bullets", [])[:3])
                if res.get("table_csv"):
                    tbls.append(res["table_csv"])
            combined = "\n\n".join(mini_summaries)
            table_csv = tbls[0] if tbls else ""
            final = self.summarize_text(combined, max_words=max_words)
            # prefer first detected table from parts
            if table_csv and not final.get("table_csv"):
                final["table_csv"] = table_csv
            return final

        user = text[:16000]  # safety on context length

        # Use JSON mode ONLY here when using ChatOpenAI
        if isinstance(self.llm, ChatOpenAI):
            llm_json = self.llm.bind(response_format={"type": "json_object"})
            out = llm_json.invoke([SystemMessage(content=SYS), HumanMessage(content=user)]).content
        else:
            out = self.llm.invoke(SYS + "\n\nCONTENT:\n" + user)

        # Robust parse: try JSON -> try to extract trailing JSON block -> fallback to prose
        payload = {"summary": out.strip(), "bullets": [], "table_csv": "", "raw": out}
        try:
            data = json.loads(out)
        except Exception:
            m = re.search(r"\{[\s\S]*\}$", out)  # last JSON-looking block
            data = json.loads(m.group(0)) if m else {}
        if isinstance(data, dict):
            payload.update(
                {
                    "summary": data.get("summary", payload["summary"]),
                    "bullets": data.get("bullets", []),
                    "table_csv": data.get("table_csv", ""),
                }
            )
        return payload
