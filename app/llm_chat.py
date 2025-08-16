# app/llm_chat.py
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

_SYS = (
    "You are a helpful analyst. Use the provided context to answer the question concisely. "
    "If the context is insufficient, say so explicitly."
)

def get_llm_response(pdf_data: dict, question: str, model_name: str = "gpt-4o-mini"):
    context = (pdf_data or {}).get("pdf_content", "")[:12000]
    msgs = [
        SystemMessage(content=_SYS),
        HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:")
    ]
    out = ChatOpenAI(model=model_name, temperature=0).invoke(msgs).content
    return {"answer": out}
