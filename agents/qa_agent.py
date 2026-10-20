"""
Retrieval-Augmented Question Answering Agent Module.
Coordinates the end-to-end RAG workflow: PDF retrieval, strict context prompting,
LLM answer generation, and source reference extraction.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Compatible import of RetrievalQA across LangChain versions
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    try:
        from langchain.chains import RetrievalQA
    except ImportError:
        from langchain_community.chains import RetrievalQA

from utils.retriever import retriever_pipeline
from utils.llm import get_llm

# Exact assignment prompt specification:
QA_PROMPT_TEMPLATE = (
    "Answer the user's question using only the provided document context. "
    "If the answer cannot be found in the uploaded document, clearly state that the "
    "information is not available in the document. Do not fabricate facts.\n\n"
    "Context:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# In-memory retriever cache to avoid re-embedding the same document across multiple questions
_RETRIEVER_CACHE: Dict[str, Any] = {}


def clear_retriever_cache():
    """Clears the in-memory cache of retrievers."""
    global _RETRIEVER_CACHE
    _RETRIEVER_CACHE.clear()


def format_sources(source_documents: List[Document]) -> str:
    """
    Formats retrieved LangChain source documents into readable citation references.

    Parameters:
        source_documents (List[Document]): Document chunks returned by the retriever.

    Returns:
        str: Clean markdown/text showing filename, page numbers, and context excerpts.
    """
    if not source_documents:
        return "No source references available for this query."

    formatted_entries = []
    seen = set()

    for idx, doc in enumerate(source_documents, start=1):
        source = doc.metadata.get("source", "Uploaded Document")
        # LangChain PyPDFLoader pages are 0-indexed; convert to 1-indexed for display
        raw_page = doc.metadata.get("page", 0)
        display_page = raw_page + 1 if isinstance(raw_page, int) else raw_page

        # Clean snippet
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        dedup_key = (source, display_page, snippet[:50])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        entry = (
            f"**[Source {len(formatted_entries) + 1}]** "
            f"`Document: {source}` | `Page: {display_page}`\n"
            f"> \"{snippet}\""
        )
        formatted_entries.append(entry)

    return "\n\n".join(formatted_entries)


def question_answering(
    file: Union[str, Path, object],
    query: str,
    retriever=None,
    llm=None,
) -> Dict[str, Any]:
    """
    Executes the Retrieval-Augmented Generation (RAG) question answering workflow.

    Workflow:
    1. Validates the query and file inputs.
    2. Retrieves or instantiates the retriever for the uploaded PDF.
    3. Retrieves relevant document chunks matching the query.
    4. Passes the retrieved context and question to the LLM with strict instructions.
    5. Generates a concise, factual answer.
    6. Returns the answer and formatted source references.

    Parameters:
        file (Union[str, Path, object]): The uploaded PDF file or file path.
        query (str): The user's question.
        retriever (Optional): An existing retriever instance (for testing or caching).
        llm (Optional): An existing LLM instance (for testing or mocking).

    Returns:
        Dict[str, Any]: {
            "answer": str,
            "sources": str,
            "source_documents": List[Document],
            "query": str
        }

    Raises:
        ValueError: If the query or file is missing or invalid.
    """
    # 1. Validate inputs
    if not query or not query.strip():
        raise ValueError("Please provide a non-empty question.")

    query_str = query.strip()

    # 2. Resolve file key for caching
    if hasattr(file, "name"):
        file_key = str(file.name)
    elif isinstance(file, (str, Path)):
        file_key = str(file)
    elif file is None and retriever is not None:
        file_key = "direct_retriever"
    else:
        raise ValueError("Please upload a valid PDF document before asking questions.")

    # 3. Obtain or create retriever
    active_retriever = retriever
    if active_retriever is None:
        if file_key in _RETRIEVER_CACHE:
            active_retriever = _RETRIEVER_CACHE[file_key]
        else:
            active_retriever = retriever_pipeline(file)
            _RETRIEVER_CACHE[file_key] = active_retriever

    # 4. Obtain or create LLM
    active_llm = llm if llm is not None else get_llm()

    # 5. Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=active_llm,
        chain_type="stuff",
        retriever=active_retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    # 6. Execute query
    try:
        response = qa_chain.invoke({"query": query_str})
    except Exception as e:
        raise RuntimeError(f"Error during RAG question answering execution: {str(e)}")

    answer = response.get("result", "").strip()
    source_docs = response.get("source_documents", [])

    # 7. Format source citations
    sources_text = format_sources(source_docs)

    return {
        "answer": answer,
        "sources": sources_text,
        "source_documents": source_docs,
        "query": query_str,
    }
