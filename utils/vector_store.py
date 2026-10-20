"""
Chroma Vector Database Module for PDF Research Assistant.
Manages persistent local embedding storage, deduplication, and similarity search.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma

from utils.embeddings import watsonx_embedding


def _generate_chunk_id(chunk: Document, source_name: str) -> str:
    """
    Generates a deterministic unique hash ID for a document chunk to prevent duplicates.
    """
    page = chunk.metadata.get("page", 0)
    chunk_idx = chunk.metadata.get("chunk_index", 0)
    # Combine source, page, chunk index, and snippet hash
    text_snippet = chunk.page_content
    raw_key = f"{source_name}_p{page}_c{chunk_idx}_{text_snippet}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]


def _sanitize_collection_name(name: str) -> str:
    """
    Sanitizes collection name to meet ChromaDB requirements:
    - 3-63 characters
    - Must start and end with an alphanumeric character
    - Contains only alphanumeric characters, underscores, or hyphens
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_-")
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}".ljust(3, "0")
    if len(cleaned) > 63:
        # Preserve hash suffix for uniqueness
        cleaned = cleaned[:50] + "_" + hashlib.md5(name.encode()).hexdigest()[:12]
    return cleaned


def vector_database(
    chunks: List[Document],
    embedding_function=None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Stores document embeddings in an isolated Chroma index; persistence is opt-in.

    Features:
    - Automatically initializes watsonx_embedding() if not provided.
    - Generates deterministic IDs to prevent duplicate indexing when the same PDF is re-uploaded.
    - Scopes the collection so only documents from the current upload/session are retrieved,
      with optional persistent on-disk storage in `persist_directory`.

    Parameters:
        chunks (List[Document]): The list of chunked Document objects.
        embedding_function: Optional embedding model (defaults to watsonx_embedding()).
        collection_name (Optional[str]): Name of the Chroma collection. Defaults to a sanitized
                                         hash based on the document source.
        persist_directory (str): Opt-in persistent storage path; None creates an isolated in-memory collection.

    Returns:
        Chroma: Persistent Chroma vector store instance ready for similarity search.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot create vector database from an empty list of chunks.")

    # 1. Resolve embedding function
    if embedding_function is None:
        embedding_function = watsonx_embedding()

    # 2. Determine source filename from metadata
    source_name = chunks[0].metadata.get("source", "uploaded_doc")
    base_name = Path(source_name).stem

    # 3. Determine collection name (session/document scoped)
    if collection_name is None:
        source_hash = hashlib.md5(source_name.encode("utf-8")).hexdigest()[:8]
        raw_name = f"doc_{base_name}_{source_hash}"
        col_name = _sanitize_collection_name(raw_name)
    else:
        col_name = _sanitize_collection_name(collection_name)

    # Content and embedding configuration isolate revisions and incompatible vectors.
    identity = repr([(c.metadata, c.page_content) for c in chunks])
    model = getattr(embedding_function, "model_id", None)
    config = (type(embedding_function).__module__, type(embedding_function).__name__,
              model, getattr(embedding_function, "size", None))
    fingerprint = hashlib.sha256((identity + repr(config)).encode()).hexdigest()[:24]
    col_name = _sanitize_collection_name(col_name[:30] + "_" + fingerprint)

    # UI uploads use a fresh in-memory collection. This avoids reusing disk
    # metadata whose HNSW segment files may be missing or held by another process.
    # Explicit persistent stores remain available to callers and tests.
    if persist_directory is None:
        from uuid import uuid4
        col_name = "session_" + uuid4().hex
    else:
        persist_directory = str(Path(persist_directory).resolve())
        os.makedirs(persist_directory, exist_ok=True)

    # 5. Generate deterministic chunk IDs to prevent duplicate indexing
    chunk_ids = [_generate_chunk_id(chunk, source_name) for chunk in chunks]

    # 6. Initialize or load the Chroma collection
    vectordb = Chroma(
        collection_name=col_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Never hide storage errors: a failed metadata read is not an empty index.
    existing_info = vectordb.get()
    existing_ids = set(existing_info.get("ids", []))

    # Filter out chunks that have already been indexed
    new_chunks = []
    new_ids = []
    for chunk, cid in zip(chunks, chunk_ids):
        if cid not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(cid)

    if new_chunks:
        vectordb.add_documents(documents=new_chunks, ids=new_ids)

    return vectordb
