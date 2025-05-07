# offline/indexer.py

import warnings
import shutil
from pathlib import Path

# silence LangChain deprecation notices
warnings.filterwarnings("ignore", category=DeprecationWarning)

# use the community packages to avoid deprecation warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from embedder import load_chunk_documents  # your loader for data/chunks

def sanitize_metadata(documents):
    """
    Ensure all metadata values are primitives (str, int, float, bool).
    Lists/tuples become comma-joined strings so Chroma will accept them.
    """
    for doc in documents:
        for key, value in list(doc.metadata.items()):
            if isinstance(value, (list, tuple)):
                doc.metadata[key] = ", ".join(str(v) for v in value)
    return documents


def create_vectorstore(
    data_dir: str,
    persist_dir: str = "db/chroma_index",
    model_name: str = "multi-qa-mpnet-base-dot-v1",
):
    # 0) remove old index if present
    idx_path = Path(persist_dir)
    if idx_path.exists():
        print(f"🗑️  Removing existing index at '{persist_dir}'")
        shutil.rmtree(persist_dir)

    # 1) Load your pre-chunked JSON docs
    docs = load_chunk_documents(data_dir)
    print(f"📄 Loaded {len(docs)} chunk Documents from '{data_dir}/chunks/'")

    # 2) Sanitize metadata
    docs = sanitize_metadata(docs)

    # 3) Embed & index
    print(f"🔗 Embedding & indexing {len(docs)} docs with '{model_name}'…")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"✅ Indexed {len(docs)} documents into Chroma at '{persist_dir}'")


if __name__ == "__main__":
    create_vectorstore("data")
