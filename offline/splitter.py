# offline/splitter.py

import json
from pathlib import Path

from loaders import load_documents
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def group_documents(docs, pages_per_chunk=3):
    """
    Batch together consecutive page-documents into multi-page Documents.
    """
    grouped = []
    for i in range(0, len(docs), pages_per_chunk):
        batch = docs[i : i + pages_per_chunk]
        # join their text
        content = "\n\n".join(d.page_content for d in batch)
        # collect their metadata if you want to trace back
        sources = [f"{d.metadata['source']} (page {d.metadata['page']})"
                   for d in batch]
        metadata = {"sources": sources}
        grouped.append(Document(page_content=content, metadata=metadata))
    return grouped

def split_documents(
    data_dir: str,
    pages_per_chunk: int = 3,
    chunk_size: int = 1000,
    chunk_overlap: int = 300
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

    print(f"Grouped into {len(grouped)} super-docs; split into {len(chunked_docs)} chunks.")
    return chunked_docs

if __name__ == "__main__":
    chunks = split_documents("data", pages_per_chunk=3)
    print(f"Total chunks created: {len(chunks)}")
