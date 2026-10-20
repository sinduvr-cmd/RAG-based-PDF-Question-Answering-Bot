"""
Comprehensive Test Suite for the RAG-Based PDF Question Answering Bot.

Covers:
- PDF document loading (valid, missing, invalid extension, empty, corrupted).
- Text splitting (chunk sizes, overlap, metadata preservation).
- watsonx.ai embeddings configuration & error handling for missing credentials.
- Chroma vector database initialization, deduplication, and persistence.
- Retriever creation and similarity search.
- Question-answering RAG pipeline execution with custom prompt and source formatting.
- Error handling for invalid inputs.
"""

import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM

from utils.document_loader import document_loader
from utils.text_splitter import text_splitter
from utils.embeddings import watsonx_embedding
from utils.vector_store import vector_database, _sanitize_collection_name
from utils.retriever import retriever_pipeline
from utils.llm import get_llm
from agents.qa_agent import (
    question_answering,
    format_sources,
    clear_retriever_cache,
    QA_PROMPT_TEMPLATE,
)

SAMPLE_PDF_PATH = Path("data/sample_research_paper.pdf")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def cleanup_cache_and_test_dirs():
    """Clear retriever cache before and after every test and clean up test Chroma DBs."""
    clear_retriever_cache()
    yield
    clear_retriever_cache()
    for test_dir in ["./test_chroma_db", "./chroma_db_test"]:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def sample_documents():
    """Returns sample LangChain documents for unit testing."""
    return [
        Document(
            page_content=(
                "The Transformer model uses an encoder-decoder structure. "
                "The encoder maps an input sequence of symbol representations to continuous representations."
            ),
            metadata={"source": "sample_paper.pdf", "page": 0},
        ),
        Document(
            page_content=(
                "Scaled Dot-Product Attention computes the dot products of the query with all keys, "
                "dividing each by the square root of d_k, and applying a softmax function."
            ),
            metadata={"source": "sample_paper.pdf", "page": 1},
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Document Loading Tests
# ---------------------------------------------------------------------------
class TestDocumentLoader:

    def test_load_valid_pdf(self):
        """Verify that a valid PDF is loaded with correct page count and metadata."""
        assert SAMPLE_PDF_PATH.exists(), "Sample test PDF must exist"
        docs = document_loader(SAMPLE_PDF_PATH)
        assert len(docs) == 3
        for idx, doc in enumerate(docs):
            assert doc.metadata["source"] == "sample_research_paper.pdf"
            assert doc.metadata["page"] == idx
            assert len(doc.page_content.strip()) > 50

    def test_missing_file_raises_error(self):
        """Verify that non-existent PDF raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
            document_loader("non_existent_file.pdf")

    def test_invalid_extension_raises_error(self, tmp_path):
        """Verify that uploading non-PDF extensions raises ValueError."""
        txt_file = tmp_path / "research.txt"
        txt_file.write_text("Some research content")
        with pytest.raises(ValueError, match="Only .pdf files are supported"):
            document_loader(txt_file)

    def test_empty_pdf_raises_error(self, tmp_path):
        """Verify that a 0-byte PDF file raises ValueError."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.touch()
        with pytest.raises(ValueError, match="is empty"):
            document_loader(empty_pdf)

    def test_corrupted_header_raises_error(self, tmp_path):
        """Verify that a file with non-PDF header raises ValueError."""
        bad_pdf = tmp_path / "corrupted.pdf"
        bad_pdf.write_bytes(b"NOT_A_PDF_HEADER_CONTENT_HERE")
        with pytest.raises(ValueError, match="not a valid PDF document"):
            document_loader(bad_pdf)

    def test_gradio_file_object_compatibility(self):
        """Verify that Gradio file objects with a .name attribute are handled correctly."""
        class MockGradioFile:
            def __init__(self, path):
                self.name = str(path)

        gradio_file = MockGradioFile(SAMPLE_PDF_PATH)
        docs = document_loader(gradio_file)
        assert len(docs) == 3


# ---------------------------------------------------------------------------
# 2. Text Splitting Tests
# ---------------------------------------------------------------------------
class TestTextSplitter:

    def test_text_splitting_and_metadata_preservation(self, sample_documents):
        """Verify that documents are split and retain source and page metadata."""
        chunks = text_splitter(sample_documents, chunk_size=80, chunk_overlap=20)
        assert len(chunks) > len(sample_documents)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert len(chunk.page_content) <= 80

    def test_empty_document_list_raises_error(self):
        """Verify that passing an empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot split an empty document list"):
            text_splitter([])

    def test_invalid_overlap_raises_error(self, sample_documents):
        """Verify that chunk_overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be strictly smaller"):
            text_splitter(sample_documents, chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------------------
# 3. Embeddings Tests
# ---------------------------------------------------------------------------
class TestEmbeddings:

    def test_missing_credentials_raises_informative_error(self, monkeypatch):
        """Verify that missing watsonx credentials raise a helpful ValueError."""
        monkeypatch.delenv("WATSONX_APIKEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        monkeypatch.delenv("WATSONX_URL", raising=False)

        with pytest.raises(ValueError, match="Missing required IBM watsonx.ai credentials"):
            watsonx_embedding()

    def test_embeddings_parameter_configuration(self):
        """Verify that WatsonxEmbeddings is configured with slate-125m and truncation parameters."""
        with patch("utils.embeddings.WatsonxEmbeddings") as mock_emb_cls:
            mock_emb_cls.return_value = MagicMock()
            emb = watsonx_embedding(
                apikey="mock_key",
                project_id="mock_project",
                url="https://us-south.ml.cloud.ibm.com",
            )
            mock_emb_cls.assert_called_once_with(
                model_id="ibm/slate-125m-english-rtrvr-v2",
                url="https://us-south.ml.cloud.ibm.com",
                apikey="mock_key",
                project_id="mock_project",
                params={"truncate_input_tokens": 512},
            )
            assert emb is not None


# ---------------------------------------------------------------------------
# 4. Chroma Vector Store Tests
# ---------------------------------------------------------------------------
class TestVectorStore:

    def test_vector_database_creation_and_deduplication(self, sample_documents):
        """Verify persistent Chroma vector store creation and duplicate prevention."""
        chunks = text_splitter(sample_documents, chunk_size=200, chunk_overlap=50)
        mock_embedding = FakeEmbeddings(size=32)
        test_dir = "./test_chroma_db"

        # 1. Initial indexing
        vectordb = vector_database(
            chunks,
            embedding_function=mock_embedding,
            collection_name="test_col_dedup",
            persist_directory=test_dir,
        )
        initial_count = vectordb._collection.count()
        assert initial_count == len(chunks)

        # 2. Re-indexing the same chunks must not increase count
        vectordb_repeat = vector_database(
            chunks,
            embedding_function=mock_embedding,
            collection_name="test_col_dedup",
            persist_directory=test_dir,
        )
        assert vectordb_repeat._collection.count() == initial_count

    def test_empty_chunks_raises_error(self):
        """Verify that empty chunks raise ValueError."""
        with pytest.raises(ValueError, match="Cannot create vector database from an empty list"):
            vector_database([])

    def test_sanitize_collection_name(self):
        """Verify Chroma collection naming rules compliance."""
        assert _sanitize_collection_name("valid_name-1") == "valid_name-1"
        assert _sanitize_collection_name("a") == "col_a"
        assert len(_sanitize_collection_name("x" * 100)) <= 63


# ---------------------------------------------------------------------------
# 5. Retriever Pipeline Tests
# ---------------------------------------------------------------------------
class TestRetrieverPipeline:

    def test_retriever_pipeline_end_to_end(self):
        """Verify that retriever_pipeline executes steps 1-5 and retrieves top-k chunks."""
        mock_embedding = FakeEmbeddings(size=32)
        retriever = retriever_pipeline(
            SAMPLE_PDF_PATH,
            embedding_function=mock_embedding,
            k=4,
        )
        assert retriever is not None
        assert retriever.search_kwargs["k"] == 4

        # Test similarity search
        query = "What is the Transformer architecture?"
        retrieved_docs = retriever.invoke(query)
        assert len(retrieved_docs) > 0
        assert len(retrieved_docs) <= 4
        assert "source" in retrieved_docs[0].metadata


# ---------------------------------------------------------------------------
# 6. LLM Configuration Tests
# ---------------------------------------------------------------------------
class TestLLMConfiguration:

    def test_missing_credentials_raises_error(self, monkeypatch):
        """Verify that missing credentials raise ValueError."""
        monkeypatch.delenv("WATSONX_APIKEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        monkeypatch.delenv("WATSONX_URL", raising=False)

        with pytest.raises(ValueError, match="Missing required IBM watsonx.ai credentials"):
            get_llm()

    def test_llm_instantiation_parameters(self):
        """Verify that WatsonxLLM receives greedy / low temperature parameters."""
        with patch("utils.llm.WatsonxLLM") as mock_llm_cls:
            mock_llm_cls.return_value = MagicMock()
            llm = get_llm(
                apikey="mock_key",
                project_id="mock_project",
                url="https://us-south.ml.cloud.ibm.com",
                model_id="ibm/granite-13b-chat-v2",
            )
            mock_llm_cls.assert_called_once()
            call_kwargs = mock_llm_cls.call_args.kwargs
            assert call_kwargs["model_id"] == "ibm/granite-13b-chat-v2"
            assert call_kwargs["params"]["decoding_method"] == "greedy"
            assert call_kwargs["params"]["temperature"] == 0.1
            assert llm is not None


# ---------------------------------------------------------------------------
# 7. Question Answering Agent Tests
# ---------------------------------------------------------------------------
class TestQuestionAnsweringAgent:

    def test_qa_agent_with_mock_llm_and_retriever(self, sample_documents):
        """Verify the full question_answering flow with custom prompt and source citations."""
        mock_embedding = FakeEmbeddings(size=32)
        vectordb = vector_database(
            sample_documents,
            embedding_function=mock_embedding,
            collection_name="test_qa_col",
            persist_directory="./test_chroma_db",
        )
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        expected_answer = "The Transformer model uses an encoder-decoder structure without recurrence."
        mock_llm = FakeListLLM(responses=[expected_answer])

        result = question_answering(
            file=SAMPLE_PDF_PATH,
            query="What structure does the Transformer use?",
            retriever=retriever,
            llm=mock_llm,
        )

        assert result["answer"] == expected_answer
        assert "sample_paper.pdf" in result["sources"]
        assert len(result["source_documents"]) > 0

    def test_prompt_template_exact_wording(self):
        """Verify that QA_PROMPT_TEMPLATE contains the exact required prompt constraint."""
        required_instruction = (
            "Answer the user's question using only the provided document context. "
            "If the answer cannot be found in the uploaded document, clearly state that the "
            "information is not available in the document. Do not fabricate facts."
        )
        assert required_instruction in QA_PROMPT_TEMPLATE

    def test_empty_query_raises_error(self):
        """Verify that empty queries raise ValueError."""
        with pytest.raises(ValueError, match="Please provide a non-empty question"):
            question_answering(file=SAMPLE_PDF_PATH, query="   ")

    def test_format_sources(self):
        """Verify format_sources produces clean citations."""
        docs = [
            Document(
                page_content="The Transformer relies on attention mechanisms.",
                metadata={"source": "test_paper.pdf", "page": 2},
            )
        ]
        formatted = format_sources(docs)
        assert "Document: test_paper.pdf" in formatted
        assert "Page: 3" in formatted  # 0-indexed page 2 converted to 1-indexed page 3
        assert "The Transformer relies on attention mechanisms." in formatted
