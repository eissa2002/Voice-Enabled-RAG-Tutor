import os
import sys
from typing import Tuple

# make sure project root is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from langchain_ollama import OllamaLLM  
from online.retrieval.retriever import get_relevant_chunks

# instantiate your local LLM
llm = OllamaLLM(model="llama3.1:8b", temperature=0)

def generate_answer(
    chunks,
    question: str,
    chat_history=None,
    target_lang: str = "en"
) -> Tuple[str, str]:
    """
    Returns (answer_text, citation_text).
    target_lang: "en" or "ar"
    chat_history: list of dicts [{"role":"user"|"bot","text":...}]
    """
    if not chunks:
        return "Sorry, I don’t know.", ""

    # Build context from chunks
    context = "\n".join(f"- {chunk.page_content.strip()}" for chunk in chunks)

    # Format chat history for prompt
    history_str = ""
    if chat_history:
        lines = []
        for turn in chat_history:
            prefix = "Student:" if turn.get("role") == "user" else "Tutor:"
            lines.append(f"{prefix} {turn.get('text')}")
        history_str = "\n".join(lines)

    # Instruction based on target language
    if target_lang == "ar":
        instr = "Answer (in Arabic only):"
    else:
        instr = "Answer (in English only):"

    # Construct prompt
    prompt_text = f"""
You are a knowledgeable AI tutor. Use only the information in the bullets below to answer the student's question.
Restate or summarize as needed, but do not introduce new concepts.

Conversation so far:
{history_str}

Reference context:
{context}

Student's new question: {question}

{instr}
""".strip()

    # Call LLM
    response = llm.generate([prompt_text])
    answer = response.generations[0][0].text.strip()

    # Build citations from chunk metadata
    sources = []
    for chunk in chunks:
        md = chunk.metadata
        if "sources" in md:
            for s in md["sources"].split(","):
                s = s.strip()
                if s and s not in sources:
                    sources.append(s)
        else:
            src = f"{md.get('source')} (page {md.get('page')})"
            if src not in sources:
                sources.append(src)
    citation_text = "\n".join(f"- {s}" for s in sources)

    return answer, citation_text


if __name__ == "__main__":
    q = "What is class recognition?"
    top_chunks = get_relevant_chunks(q, top_k=5)
    ans, cites = generate_answer(top_chunks, q)
    print("\nAnswer:\n", ans)
    print("\nSources:\n", cites)
