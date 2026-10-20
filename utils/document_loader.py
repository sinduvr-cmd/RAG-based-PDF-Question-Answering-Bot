"""
Document Loader Module for Quest Analytics – Research Document AI Assistant.
Task 1: Loads documents using LangChain loaders (PyPDFLoader for PDF, TextLoader for text files).
Extracts text from every page, preserves metadata, and performs robust validation.
"""

import os
import shutil
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def text_file_loader(file_path: Union[str, Path]) -> List[Document]:
    """
    Demonstrates an additional LangChain document source using TextLoader.
    Loads local text (.txt) files and returns LangChain Document objects.

    Parameters:
        file_path (Union[str, Path]): Path to the local text file.

    Returns:
        List[Document]: List of LangChain Document objects with source metadata.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file does not exist: {file_path}")

    # Use LangChain TextLoader with explicit UTF-8 encoding
    loader = TextLoader(str(path_obj), encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = path_obj.name
        doc.metadata["page"] = 0

    return documents


def document_loader(file: Union[str, Path, object], allow_txt: bool = False) -> List[Document]:
    """
    Loads and extracts text and metadata from an uploaded research document.
    Uses LangChain's PyPDFLoader for PDF files (and TextLoader for text files when allow_txt=True).

    Parameters:
        file (Union[str, Path, object]): File path string, Path object, or Gradio file object
                                          (which contains a `.name` attribute).
        allow_txt (bool): If True, allows loading plain text files via TextLoader. Default False.

    Returns:
        List[Document]: A list of LangChain Document objects containing page content
                        and preserved metadata (source filename and page number).

    Raises:
        FileNotFoundError: If the provided file does not exist on disk.
        ValueError: If the file is invalid, empty (0 bytes), has no extractable text,
                    or is corrupted.
    """
    # 1. Resolve file path from Path, string, or Gradio file object
    if isinstance(file, Path):
        file_path = str(file.resolve())
    elif isinstance(file, str):
        file_path = os.path.abspath(file)
    elif hasattr(file, "name"):
        file_path = str(file.name)
    else:
        raise ValueError(
            f"Unsupported file input type: {type(file)}. Expected file path or file object."
        )

    # 2. Check if file exists
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file does not exist: {file_path}")

    # 3. Check file extension (Validate PDF as primary requirement)
    suffix = path_obj.suffix.lower()
    if suffix == ".txt" and allow_txt:
        return text_file_loader(path_obj)
    elif suffix != ".pdf":
        raise ValueError(
            f"Invalid file format '{path_obj.suffix}'. Only .pdf files are supported."
        )

    # 4. Check file size (must be non-empty)
    file_size = path_obj.stat().st_size
    if file_size == 0:
        raise ValueError("The provided PDF file is empty (0 bytes).")

    # 5. Check PDF magic header (%PDF-)
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF-"):
                raise ValueError(
                    "The file is not a valid PDF document (missing standard '%PDF-' header)."
                )
    except IOError as e:
        raise ValueError(f"Unable to read the PDF file: {e}")

    # 6. Copy file to project data/uploads directory for record-keeping
    try:
        project_root = Path(__file__).resolve().parent.parent
        uploads_dir = project_root / "data" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest_path = uploads_dir / path_obj.name
        if path_obj.resolve() != dest_path.resolve():
            shutil.copy2(file_path, dest_path)
    except Exception:
        pass

    # 7. Load PDF pages using LangChain's PyPDFLoader
    try:
        # PyPDFLoader parses page by page, preserving page numbering in document metadata
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {str(e)}")

    # 8. Check that pages were loaded
    if not documents:
        raise ValueError("The PDF contains no pages or could not be parsed.")

    # 9. Verify that pages contain extractable text (avoid scanned image PDFs without OCR)
    total_text = "".join(doc.page_content.strip() for doc in documents)
    if not total_text:
        raise ValueError(
            "The PDF contains no extractable text. The file might be a scanned image "
            "without an OCR text layer."
        )

    # 10. Normalize metadata: ensure 'source' uses the clean filename and 'page' exists
    for doc in documents:
        doc.metadata["source"] = path_obj.name
        if "page" not in doc.metadata:
            doc.metadata["page"] = 0

    return documents
