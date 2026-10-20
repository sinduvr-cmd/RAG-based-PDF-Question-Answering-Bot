"""
Document Embeddings Module for Quest Analytics – Research Document AI Assistant.
Task 3: Converts text chunks into dense vector representations for semantic retrieval.

HOW EMBEDDINGS WORK IN RAG:
Text embeddings map textual information (sentences, paragraphs, or chunks) into a high-dimensional
continuous vector space (e.g., 384 dimensions). Words and passages that share semantic meaning
cluster closely together in this geometric space, even if they use completely different vocabulary.
During retrieval, cosine similarity between the query vector and chunk vectors enables the system
to locate the most conceptually relevant passages for the LLM.
"""

import os
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_ibm import WatsonxEmbeddings

# Load environment variables from .env
load_dotenv()

# Embedding constants
DEFAULT_WATSONX_EMBEDDING_MODEL = "ibm/slate-125m-english-rtrvr-v2"
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_local_embeddings(model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL):
    """
    Initializes a local Sentence Transformers embedding model via LangChain.
    Used when IBM watsonx cloud embeddings are unconfigured or unavailable.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def watsonx_embedding(
    model_id: Optional[str] = None,
    apikey: Optional[str] = None,
    project_id: Optional[str] = None,
    url: Optional[str] = None,
):
    """
    Initializes an IBM watsonx.ai embedding instance.
    Uses 'ibm/slate-125m-english-rtrvr-v2' with token truncation.

    Credentials:
        WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL
    """

    resolved_apikey = apikey or os.getenv("WATSONX_APIKEY")
    resolved_project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
    resolved_url = url or os.getenv("WATSONX_URL")
    resolved_model_id = (
        model_id
        or os.getenv("WATSONX_EMBEDDING_MODEL_ID")
        or DEFAULT_WATSONX_EMBEDDING_MODEL
    )

    missing = []
    if not resolved_apikey:
        missing.append("WATSONX_APIKEY")
    if not resolved_project_id:
        missing.append("WATSONX_PROJECT_ID")
    if not resolved_url:
        missing.append("WATSONX_URL")

    if missing:
        raise ValueError(
            f"Missing required IBM watsonx.ai credentials: {', '.join(missing)}.\n"
            "Please configure them in your .env file."
        )

    embedding_params = {
        "truncate_input_tokens": 512,
    }

    return WatsonxEmbeddings(
        model_id=resolved_model_id,
        url=resolved_url,
        apikey=resolved_apikey,
        project_id=resolved_project_id,
        params=embedding_params,
    )


def get_embeddings_model(prefer_watsonx: bool = True):
    """
    Primary embedding factory:
    Attempts to initialize IBM watsonx embeddings if credentials are provided and valid;
    gracefully falls back to local Sentence Transformers model ('all-MiniLM-L6-v2') otherwise.
    """
    if prefer_watsonx:
        try:
            emb = watsonx_embedding()
            # Quick validation ping to confirm account / WML association
            emb.embed_query("ping")
            return emb, "IBM watsonx (ibm/slate-125m-english-rtrvr-v2)"
        except Exception:
            # Fall back to local Sentence Transformers as permitted by assignment rubric
            pass

    local_emb = get_local_embeddings()
    return local_emb, f"Local Sentence Transformers ({DEFAULT_LOCAL_EMBEDDING_MODEL})"


def validate_embedding_output(embedding_model, sample_text: str = "Quest Analytics RAG Assistant") -> Dict[str, Any]:
    """
    Validates that the embedding model produces non-empty float vector representations.
    Returns embedding vector dimension and sample confirmation.
    """
    vector = embedding_model.embed_query(sample_text)
    if not vector or not isinstance(vector, list):
        raise ValueError("Embedding model returned empty or invalid vector structure.")

    dimension = len(vector)
    return {
        "valid": True,
        "sample_text": sample_text,
        "embedding_dimension": dimension,
        "sample_vector_preview": vector[:5],
    }
