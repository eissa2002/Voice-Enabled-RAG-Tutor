# online/retrieval/retriever.py

import warnings
# Silence all warnings (including LangChain deprecation warnings)
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_relevant_chunks(
    query: str,
    persist_dir: str = "db/chroma_index",
    model_name: str = "multi-qa-mpnet-base-dot-v1",
    top_k: int = 3,
    min_score: float = 0.0  # only keep chunks with score ≥ this threshold
):
    """
    Given a text query, load your local Chroma index and return the top_k
    most similar Document chunks whose similarity score ≥ min_score.
    """
    # 1) Initialize the same embedding model you used offline
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2) Load your persisted Chroma store
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # 3) Perform similarity search with scores (cosine similarity by default)
    results = vectordb.similarity_search_with_score(query, k=top_k)

    # 4) Filter out chunks below the min_score threshold
    filtered_docs = [doc for doc, score in results if score >= min_score]
    return filtered_docs


if __name__ == "__main__":
    # Quick test with adjustable minimum score
    hits = get_relevant_chunks(
        "What is the class recognition?",
        top_k=5,
        min_score=0.1  # tune as needed
    )
    for doc in hits:
        print("—", doc.metadata, "\n", doc.page_content[:500].replace("\n", " "), "\n")
