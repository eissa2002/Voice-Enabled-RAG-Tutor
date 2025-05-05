# online/retrieval/retriever.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_relevant_chunks(
    query: str,
    persist_dir: str = "db/chroma_index",
    model_name: str = "multi-qa-mpnet-base-dot-v1",
    top_k: int = 3
):
    """
    Given a text query, load your local Chroma index and return the top_k
    most similar Document chunks using cosine similarity (Chroma default).
    """
    # 1) Initialize the same embedding model you used offline
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2) Load your persisted Chroma store
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # 3) Perform similarity search (cosine by default)
    docs = vectordb.similarity_search(query, k=top_k)
    return docs

if __name__ == "__main__":
    # Quick test
    hits = get_relevant_chunks("What is the ratio test in feature matching?", top_k=5)
    for doc in hits:
        print("—", doc.metadata, "\n", doc.page_content[:200].replace("\n"," "), "\n")
