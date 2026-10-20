"""
Text Splitter Module for PDF Research Assistant.
Splits loaded LangChain Document objects into manageable chunks for vector embedding.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def text_splitter(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Divides large documents into smaller overlapping text chunks while preserving metadata.

    Parameters:
        documents (List[Document]): The list of Document objects extracted by document_loader.
        chunk_size (int): The maximum character length of each chunk (default: 1000).
        chunk_overlap (int): The character overlap between consecutive chunks (default: 200).
        separators (Optional[List[str]]): Custom separator list for splitting.

    Returns:
        List[Document]: A list of chunked Document objects with preserved metadata.

    Raises:
        ValueError: If documents input is empty or invalid.
    """
    if not documents:
        raise ValueError("Cannot split an empty document list. Provide at least one valid Document.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})."
        )

    # Use standard hierarchy of separators: paragraphs -> sentences -> words -> characters
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting resulted in zero chunks.")

    # Preserve and enrich metadata: add chunk index for tracing
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    return chunks
