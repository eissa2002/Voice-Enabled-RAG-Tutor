# online/llm/inference.py

import os
import sys
from typing import Tuple

# make sure project root is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from langchain_community.llms import Ollama
from online.retrieval.retriever import get_relevant_chunks

# instantiate your local LLM
llm = Ollama(model="llama3.2", temperature=0)

def generate_answer(
    chunks,
    question: str
) -> Tuple[str, str]:
    """
    Returns (answer_text, citation_text).
    """
    if not chunks:
        return "Sorry, I don’t know.", ""

    # 1) Build a bullet-list context from the chunks
    context = "\n".join(f"- {chunk.page_content.strip()}" for chunk in chunks)

    # 2) Put that into your prompt
    prompt_text = f"""
You are a knowledgeable AI tutor. Answer the student’s question using
only the information in the bullets below. You may restate or summarize in your own words,
but do not introduce new concepts.

Context:
{context}

Question: {question}

Answer (in English only):
""".strip()

    # 3) Call the LLM
    response = llm.generate([prompt_text])
    answer = response.generations[0][0].text.strip()

    # 4) Build citation list
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
