"""
Document Retriever Module for Quest Analytics – Research Document AI Assistant.
Task 5: Converts the Chroma vector database into a semantic similarity retriever.
Configured with top-4 retrieval (k=4) and metadata preservation.
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from utils.document_loader import document_loader
from utils.text_splitter import text_splitter
from utils.embeddings import get_embeddings_model
from utils.vector_store import vector_database


def retriever_pipeline(
    file: Union[str, Path, object],
    embedding_function=None,
    k: int = 4,
    persist_directory: Optional[str] = "./chroma_db",
) -> VectorStoreRetriever:
    """
    Constructs the end-to-end LangChain similarity retriever pipeline from an uploaded document:

    1. Loads the document via `document_loader(file)` using PyPDFLoader or TextLoader.
    2. Splits document pages into overlapping chunks using `text_splitter(documents)`.
    3. Resolves embeddings (IBM watsonx or local Sentence Transformers).
    4. Initializes persistent Chroma vector database with duplicate chunk prevention.
    5. Converts the vector database to a retriever with `search_type="similarity"` and `k=4`.

    Parameters:
        file (Union[str, Path, object]): Document path or Gradio file object.
        embedding_function: Optional embedding model instance.
        k (int): Number of top similar chunks to retrieve (default: 4).
        persist_directory (Optional[str]): Local Chroma persistence directory.

    Returns:
        VectorStoreRetriever: Configured similarity retriever.
    """
    # Step 1: Load the PDF document
    documents = document_loader(file)

    # Step 2: Split documents into manageable chunks
    chunks = text_splitter(documents)

    # Step 3: Configure embeddings model
    if embedding_function is None:
        embedding_function, _ = get_embeddings_model()

    # Step 4: Create/load the persistent Chroma vector database
    vectordb = vector_database(
        chunks=chunks,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Step 5: Convert vector database to retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return retriever


def debug_retrieve_chunks(retriever: VectorStoreRetriever, query: str) -> List[Dict[str, Any]]:
    """
    Invokes the retriever with a test query and returns structured inspection details
    including total chunks retrieved, source filenames, page citations, and text previews.
    """
    results = retriever.invoke(query)
    debug_info = []
    for idx, doc in enumerate(results, start=1):
        debug_info.append({
            "rank": idx,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "content_preview": doc.page_content[:180].replace("\n", " ") + "...",
        })
    return debug_info
