# app/rag_graph_pinecone.py
import os
from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from app.pinecone_embeds import query_pinecone, INDEX_NAME, pc
from app.llm_chat import get_llm_response

load_dotenv()

class RAGState(TypedDict, total=False):
    question: str
    top_k: Optional[int]
    rag_output: str

def rag_agent(state: RAGState) -> Dict[str, Any]:
    query = state.get("question", "Summarize NVIDIA's performance.")
    user_top_k = state.get("top_k", 500)

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    total_records = stats.get("total_vector_count", 0)

    actual_top_k = min(user_top_k, total_records, 500)

    results = query_pinecone(query_text=query, top_k=actual_top_k)

    chunks = [
        m.get("metadata", {}).get("text", "") or m.get("text", "")
        for m in results.get("matches", [])
    ]
    context = " ".join(chunks)

    if not context.strip():
        return {"rag_output": "No relevant content found in Pinecone index."}

    pdf_data = {"pdf_content": context, "tables": []}
    response = get_llm_response(pdf_data, query, "gpt-4o-mini")

    state["rag_output"] = response["answer"]
    return state

def build_graph():
    builder = StateGraph(RAGState)
    builder.add_node("RAGAgent", RunnableLambda(rag_agent))
    builder.set_entry_point("RAGAgent")
    builder.add_edge("RAGAgent", END)
    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    sample_state = {"question": "What is the Summary of Significant Accounting Policies?", "top_k": 500}
    result = graph.invoke(sample_state)
    print("\n RAG Agent Output:\n")
    print(result.get("rag_output"))
