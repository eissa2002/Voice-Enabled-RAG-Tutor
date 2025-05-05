# online/llm/inference.py

import os
import sys

# Ensure project root is on sys.path for imports
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, project_root)

from langchain_community.llms import Ollama
from online.retrieval.retriever import get_relevant_chunks

# Instantiate your local LLM
llm = Ollama(model="llama3.2", temperature=0)

def generate_answer(chunks, question: str) -> str:
    """
    Given a list of Document chunks and a question string, returns a grounded answer
    and appends citation of where the key information was found.
    """
    # Build context string
    context = "\n\n".join(chunk.page_content for chunk in chunks)

    # Construct a balanced prompt
    prompt_text = f"""
You are a knowledgeable computer-vision tutor. Answer the student’s question using
only the information in the bullets below. You may restate or summarize in your own words,
but do not introduce new concepts.

Context:
{context}

Question: {question}

Answer:"
    """.strip()

    # Print prompt for debugging
    print("=== PROMPT ===")
    print(prompt_text)
    print("==============")

    # Call LLM with a single-item prompt list
    response = llm.generate([prompt_text])
    answer = response.generations[0][0].text.strip()

    # Build citations from metadata
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

    # Append citations to answer
    citation_text = "\n".join(f"- {s}" for s in sources)
    return f"{answer}\n\nSources:\n{citation_text}"

if __name__ == "__main__":
    q = "can you explain the Local receptive field:"
    top_chunks = get_relevant_chunks(q, top_k=3)
    result = generate_answer(top_chunks, q)
    print("\nAnswer:\n", result)
