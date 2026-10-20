"""
Retriever Pipeline Module for PDF Research Assistant.
Assembles document loading, text splitting, embeddings generation, and vector store retrieval.
"""

from typing import Union, Optional
from pathlib import Path
from langchain_core.vectorstores import VectorStoreRetriever

from utils.document_loader import document_loader
from utils.text_splitter import text_splitter
from utils.embeddings import watsonx_embedding
from utils.vector_store import vector_database


def retriever_pipeline(
    file: Union[str, Path, object],
    embedding_function=None,
    k: int = 4,
) -> VectorStoreRetriever:
    """
    Constructs an end-to-end LangChain retriever pipeline from an uploaded PDF file:

    1. Loads the PDF using `document_loader(file)`.
    2. Splits document pages into overlapping chunks using `text_splitter(documents)`.
    3. Configures IBM watsonx embeddings using `watsonx_embedding()`.
    4. Creates/loads the persistent Chroma vector store with duplicate prevention.
    5. Converts the vector store into a similarity retriever with top-k retrieval.

    Parameters:
        file (Union[str, Path, object]): PDF file path or Gradio file object.
        embedding_function: Optional embedding model (defaults to watsonx_embedding()).
        k (int): Number of top similar chunks to retrieve (default: 4).

    Returns:
        VectorStoreRetriever: Configured similarity retriever.
    """
    # Step 1: Load the PDF document
    documents = document_loader(file)

    # Step 2: Split documents into manageable chunks
    chunks = text_splitter(documents)

    # Step 3: Configure embeddings
    if embedding_function is None:
        embedding_function = watsonx_embedding()

    # Step 4: Create/update the vector database
    vectordb = vector_database(chunks, embedding_function=embedding_function)

    # Step 5: Convert the vector database into a retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return retriever
