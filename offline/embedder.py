# offline/embedder.py

import json
from pathlib import Path
from langchain.schema import Document

def load_chunk_documents(data_dir: str):
    """
    Reads every chunk_NNNN.json from data/chunks, and
    returns a list of Document(page_content, metadata) objects.
    """
    chunks_dir = Path(data_dir) / "chunks"
    docs = []
    for chunk_file in sorted(chunks_dir.glob("chunk_*.json")):
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        docs.append(
            Document(
                page_content=payload["page_content"],
                metadata=payload["metadata"]
            )
        )
    return docs
