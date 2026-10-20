"""
Text Splitter Module for Quest Analytics – Research Document AI Assistant.
Task 2: Splits loaded LangChain Document objects into manageable chunks using RecursiveCharacterTextSplitter.

WHY TEXT SPLITTING IS CRITICAL FOR RAG:
1. LLM Context Limits: LLMs have finite input token context windows. Passing an entire 50-page PDF
   is computationally expensive and often exceeds prompt token limits.
2. Semantic Precision: Dense retrieval algorithms (e.g. cosine similarity over embeddings) work best
   when comparing a concise, focused passage against a user question, rather than diluting meaning across an entire document.
3. Chunk Overlap: The 200-character overlap prevents sentences and thoughts from being severed arbitrarily
   at chunk boundaries, ensuring continuous context across consecutive chunks.
"""

from typing import List, Optional, Dict, Any
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
        separators (Optional[List[str]]): Custom separator list for hierarchical splitting.

    Returns:
        List[Document]: A list of chunked Document objects with preserved metadata.

    Raises:
        ValueError: If documents input is empty or chunk_overlap >= chunk_size.
    """
    if not documents:
        raise ValueError("Cannot split an empty document list. Provide at least one valid Document.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})."
        )

    # Standard natural hierarchy of split separators:
    # 1. Paragraphs ("\n\n")
    # 2. Lines ("\n")
    # 3. Spaces (" ")
    # 4. Individual characters ("")
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


def inspect_first_chunk(chunks: List[Document]) -> Dict[str, Any]:
    """
    Utility function for inspection during testing and demonstrations:
    Returns the total chunk count and metadata/content preview of the first chunk.
    """
    if not chunks:
        return {"total_chunks": 0, "first_chunk": None}

    first = chunks[0]
    return {
        "total_chunks": len(chunks),
        "first_chunk_metadata": first.metadata,
        "first_chunk_length": len(first.page_content),
        "first_chunk_preview": first.page_content[:250] + ("..." if len(first.page_content) > 250 else ""),
    }
