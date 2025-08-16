# # app/prompts.py

# """
# Prompt templates and helpers used by the RAG pipeline.
# """

# from langchain.prompts import PromptTemplate

# # System behavior (kept short so it fits in context every time)
# SYSTEM_INSTRUCTIONS = (
#     "You are a precise technical assistant for data/ETL/SQL/Snowflake docs. "
#     "Only use the provided context. If the context is insufficient, say "
#     "\"Insufficient context.\" and suggest what to ingest. Prefer concise, "
#     "actionable steps and runnable code. Always add a 'Sources:' section."
# )

# # What the model actually sees for each question
# ANSWER_PROMPT = PromptTemplate.from_template(
#     """Question:
# {question}

# Context (use ONLY this; do not invent facts):
# {context}

# Answer requirements:
# - Start with a brief, precise answer (2–5 sentences).
# - If helpful, include step-by-step instructions or code blocks.
# - End with:
#   Sources:
#   - <filename or title> (and page/line if available)

# If the context does not contain the answer, reply exactly:
# Insufficient context.
# Then add a short note about which doc(s) to ingest next.
# """
# )

# app/prompts.py

from langchain.prompts import PromptTemplate

SYSTEM_INSTRUCTIONS = (
    "You are a precise technical assistant for data/ETL/SQL/Snowflake docs. "
    "Only use the provided context. If the context is insufficient, say "
    '"Insufficient context." and suggest what to ingest next. Prefer concise, '
    "actionable steps and runnable code. Do NOT include a 'Sources:' section; "
    "the system will add sources automatically."
)

ANSWER_PROMPT = PromptTemplate.from_template(
    """Question:
{question}

Context (use ONLY this; do not invent facts):
{context}

Answer requirements:
- Start with a brief, precise answer (2–5 sentences).
- If helpful, include step-by-step instructions or code blocks.
- If the context does not contain the answer, reply exactly:
Insufficient context.
Then add a short note about which doc(s) to ingest next.
"""
)


def build_messages(question: str, context: str):
    """
    Returns chat messages in (role, content) tuples compatible with many LLM wrappers.
    """
    return [
        ("system", SYSTEM_INSTRUCTIONS),
        ("user", ANSWER_PROMPT.format(question=question, context=context)),
    ]
