# app/chunkers.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def default_splitter(chunk_size=900, chunk_overlap=120):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def split_docs(docs, chunk_size=900, chunk_overlap=120):
    splitter = default_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)
    # normalize missing metadata
    out = []
    for d in chunks:
        meta = d.metadata or {}
        meta.setdefault(
            "source", meta.get("file_path") or meta.get("source") or "unknown"
        )
        out.append(Document(page_content=d.page_content, metadata=meta))
    return out
