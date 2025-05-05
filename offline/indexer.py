# offline/indexer.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

from embedder import load_chunk_documents


def sanitize_metadata(documents):
    """
    Ensure all metadata values are primitives (str, int, float, bool).
    Lists and other complex types are converted to comma-separated strings.
    """
    for doc in documents:
        for key, value in list(doc.metadata.items()):
            if isinstance(value, (list, tuple)):
                # Convert list items to strings
                doc.metadata[key] = ", ".join(str(item) for item in value)
            # You can add more type checks here if needed
    return documents


def create_vectorstore(
    data_dir: str,
    persist_dir: str = "db/chroma_index",
    model_name: str = "multi-qa-mpnet-base-dot-v1"
):
    """
    Loads chunk Documents, sanitizes metadata, embeds them with a HuggingFace model,
    and persists into a Chroma vector store.
    """
    # 1) Load all chunk Documents
    docs = load_chunk_documents(data_dir)
    print(f"Loaded {len(docs)} chunk Documents.")

    # 2) Sanitize metadata to avoid non-primitive values
    docs = sanitize_metadata(docs)

    # 3) Embed with specified HuggingFace model
    print(f"Embedding and indexing {len(docs)} chunks using '{model_name}'...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 4) Create or load Chroma vector store
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"Indexed {len(docs)} documents into Chroma at '{persist_dir}'")


if __name__ == "__main__":
    create_vectorstore("data")
