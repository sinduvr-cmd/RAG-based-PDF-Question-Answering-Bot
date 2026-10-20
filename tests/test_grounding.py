import json
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.embeddings import DeterministicFakeEmbedding
from agents.qa_agent import question_answering, format_sources
from utils.local_retrieval import LocalRetriever
from utils.vector_store import vector_database

def test_offline_retrieval_and_abstention():
    docs = [Document(page_content="Orion-01: MAE 0.35 mg/L; single laboratory only.", metadata={"source": "research.pdf", "page": 1}),
            Document(page_content="Delta-01: MAE 0.42 mg/L; low-turbidity water only.", metadata={"source": "research.pdf", "page": 1})]
    retriever = LocalRetriever(docs)
    result = question_answering(None, "Which focal record has the lowest listed MAE, and what is its limitation?", retriever, offline=True)
    assert "0.35 mg/L" in result["answer"]
    assert "single laboratory only" in result["answer"]
    assert "Page: 2" in result["sources"]
    assert "Offline local retrieval" in result["answer"]
    missing = question_answering(None, "Who won the Nobel prize?", retriever, offline=True)
    assert "not available" in missing["answer"]
    assert not missing["source_documents"]

def test_reject_fabricated_quote_and_page():
    retriever = LocalRetriever([Document(page_content="MAE is 0.35 mg/L.", metadata={"page": 1})])
    for item in [{"source": 1, "quote": "MAE is 9 mg/L."}, {"source": 99, "quote": "MAE is 0.35 mg/L."}]:
        llm = FakeListLLM(responses=[json.dumps({"evidence": [item]})])
        result = question_answering(None, "What MAE?", retriever, llm)
        assert "not available" in result["answer"]
        assert not result["source_documents"]
    assert "Page: Unknown" in format_sources([Document(page_content="text")])

def test_content_revision_isolation(tmp_path):
    embedding = DeterministicFakeEmbedding(size=32)
    original = Document(page_content="x" * 120 + " old value", metadata={"source": "same.pdf", "page": 0})
    revision = Document(page_content="x" * 120 + " new value", metadata=original.metadata)
    first = vector_database([original], embedding, "revision", str(tmp_path))
    second = vector_database([revision], embedding, "revision", str(tmp_path))
    assert first._collection.name != second._collection.name
    assert second.get()["documents"] == [revision.page_content]

def test_failed_upload_clears_previous_document():
    from app import process_document
    status, state = process_document(None, True, {"retriever": "old"})
    assert state == {}


def test_chat_json_with_trailing_prose():
    doc = Document(page_content="Orion MAE is 0.35 mg/L.", metadata={"page": 1})
    retriever = LocalRetriever([doc])
    for suffix in ["", " Additional model prose must never be shown."]:
        llm = FakeListLLM(responses=[json.dumps({"evidence": [{"source": 1, "quote": doc.page_content}]}) + suffix])
        result = question_answering(None, "What MAE?", retriever, llm)
        assert result["answer"] == doc.page_content + " [Source 1]"
    llm = FakeListLLM(responses=['{"evidence": []} No answer found.'])
    result = question_answering(None, "What MAE?", retriever, llm)
    assert "not available" in result["answer"]
    assert "model did not" not in result["answer"]


def test_default_upload_index_isolated_and_searchable():
    docs = [Document(page_content="Orion MAE 0.35 mg/L", metadata={"page": 1})]
    embedding = DeterministicFakeEmbedding(size=32)
    first = vector_database(docs, embedding)
    second = vector_database(docs, embedding)
    assert first._collection.name != second._collection.name
    assert first._client.get_settings().is_persistent is False
    assert second.similarity_search("Orion MAE", k=1)[0].page_content == docs[0].page_content
    assert first.similarity_search("Orion MAE", k=1)[0].metadata["page"] == 1
