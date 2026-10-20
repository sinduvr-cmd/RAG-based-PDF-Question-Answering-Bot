"""
Document Loader Module for PDF Research Assistant.
Responsible for extracting text and metadata from uploaded PDF documents.
"""

import os
import shutil
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def document_loader(file: Union[str, Path, object]) -> List[Document]:
    """
    Loads and extracts text and metadata from an uploaded PDF file.

    Parameters:
        file (Union[str, Path, object]): File path string, Path object, or Gradio file object
                                          (which contains a `.name` attribute).

    Returns:
        List[Document]: A list of LangChain Document objects containing page content
                        and metadata (source filename and page number).

    Raises:
        FileNotFoundError: If the provided file does not exist on disk.
        ValueError: If the file is not a PDF, is empty (0 bytes), has no extractable text,
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

    # 3. Check file extension
    if path_obj.suffix.lower() != ".pdf":
        raise ValueError(
            f"Invalid file format '{path_obj.suffix}'. Only .pdf files are supported."
        )

    # 4. Check file size (non-empty)
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

    # 6. Copy file to project data/uploads directory for record-keeping if not already there
    try:
        project_root = Path(__file__).resolve().parent.parent
        uploads_dir = project_root / "data" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest_path = uploads_dir / path_obj.name
        if path_obj.resolve() != dest_path.resolve():
            shutil.copy2(file_path, dest_path)
    except Exception:
        # Non-critical: continue processing from original file location
        pass

    # 7. Load PDF pages using LangChain's PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {str(e)}")

    # 8. Check that pages were loaded
    if not documents:
        raise ValueError("The PDF contains no pages or could not be parsed.")

    # 9. Verify that pages contain extractable text
    total_text = "".join(doc.page_content.strip() for doc in documents)
    if not total_text:
        raise ValueError(
            "The PDF contains no extractable text. The file might be a scanned image "
            "without an OCR text layer."
        )

    # 10. Normalize metadata: ensure 'source' uses the base filename for clean display
    for doc in documents:
        # Keep filename as clean source identifier
        doc.metadata["source"] = path_obj.name
        # PyPDFLoader pages are 0-indexed; ensure page exists
        if "page" not in doc.metadata:
            doc.metadata["page"] = 0

    return documents
