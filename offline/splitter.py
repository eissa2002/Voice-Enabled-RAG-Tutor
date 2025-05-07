# offline/splitter.py

import json
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loaders import load_documents  # your existing loader

def group_documents(docs, pages_per_chunk=1):
    """
    Batch together consecutive page-documents into multi-page Documents.
    With slides, we want 1 slide per chunk.
    """
    grouped = []
    for i in range(0, len(docs), pages_per_chunk):
        batch = docs[i : i + pages_per_chunk]
        content = "\n\n".join(d.page_content for d in batch)
        # keep the same metadata grouping
        sources = [f"{d.metadata['source']} (page {d.metadata.get('page', d.metadata.get('slide_number', '?'))})"
                   for d in batch]
        grouped.append(Document(page_content=content, metadata={"sources": sources}))
    return grouped

def split_documents(
    data_dir: str,
    pages_per_chunk: int = 1,     # one slide per super-doc
    chunk_size: int = 600,        # ~600 chars ≈ 1–2 paragraphs
    chunk_overlap: int = 120      # overlap by ~120 chars (~1–2 sentences)
):
    # 1. Load every single-page Document
    docs = load_documents(data_dir)

    # 2. Group N pages into one bigger Document
    grouped = group_documents(docs, pages_per_chunk=pages_per_chunk)

    # 3. Split each grouped Document into character-based chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = splitter.split_documents(grouped)

    # 4. Persist
    chunks_dir = Path(data_dir) / "chunks"
    chunks_dir.mkdir(exist_ok=True, parents=True)
    for idx, doc in enumerate(chunked_docs):
        payload = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        (chunks_dir / f"chunk_{idx:04d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print(f"Grouped into {len(grouped)} slide-docs; split into {len(chunked_docs)} chunks.")
    return chunked_docs

if __name__ == "__main__":
    chunks = split_documents("data")
    print(f"Total chunks created: {len(chunks)}")
