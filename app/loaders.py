# app/loaders.py
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

CODE_EXTS = {".py", ".sql", ".yaml", ".yml", ".json", ".md"}
TEXT_EXTS = {".txt", ".md"}
PDF_EXTS = {".pdf"}


def load_docs(input_dir: str):
    """Load text, code, and PDFs into LangChain Documents."""
    docs = []
    for p in Path(input_dir).rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext in TEXT_EXTS or ext in CODE_EXTS:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif ext in PDF_EXTS:
                docs.extend(PyMuPDFLoader(str(p)).load())
        except Exception as e:
            print(f"[loader] Skipped {p}: {e}")
    return docs
