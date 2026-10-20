"""
IBM watsonx.ai Embeddings Module for PDF Research Assistant.
Initializes the WatsonxEmbeddings client using the slate-125m dense retrieval model.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_ibm import WatsonxEmbeddings

# Load environment variables from .env file
load_dotenv()


def watsonx_embedding(
    model_id: Optional[str] = None,
    apikey: Optional[str] = None,
    project_id: Optional[str] = None,
    url: Optional[str] = None,
) -> WatsonxEmbeddings:
    """
    Initializes and returns an IBM watsonx.ai embedding instance.

    Uses the dense retrieval model: 'ibm/slate-125m-english-rtrvr-v2' with
    input token truncation enabled.

    Credentials are read from environment variables:
        WATSONX_APIKEY
        WATSONX_PROJECT_ID
        WATSONX_URL

    Parameters:
        model_id (Optional[str]): Watsonx embedding model ID (defaults to WATSONX_EMBEDDING_MODEL_ID
                                   or 'ibm/slate-125m-english-rtrvr-v2').
        apikey (Optional[str]): Explicit API key (defaults to WATSONX_APIKEY env var).
        project_id (Optional[str]): Explicit Project ID (defaults to WATSONX_PROJECT_ID env var).
        url (Optional[str]): Explicit Watsonx service URL (defaults to WATSONX_URL env var).

    Returns:
        WatsonxEmbeddings: Configured LangChain embedding model for watsonx.ai.

    Raises:
        ValueError: If any required IBM watsonx.ai credentials are missing or invalid.
    """
    # 1. Resolve credentials from arguments or environment variables
    resolved_apikey = apikey or os.getenv("WATSONX_APIKEY")
    resolved_project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
    resolved_url = url or os.getenv("WATSONX_URL")
    resolved_model_id = (
        model_id
        or os.getenv("WATSONX_EMBEDDING_MODEL_ID")
        or "ibm/slate-125m-english-rtrvr-v2"
    )

    # 2. Validate presence of credentials
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
            "Please configure them in your .env file or export them as environment variables.\n"
            "Refer to .env.example for instructions on obtaining these credentials."
        )

    # 3. Configure truncation parameters supported by slate-125m (max 512 tokens)
    embedding_params = {
        "truncate_input_tokens": 512,
    }

    # 4. Initialize WatsonxEmbeddings
    try:
        embeddings = WatsonxEmbeddings(
            model_id=resolved_model_id,
            url=resolved_url,
            apikey=resolved_apikey,
            project_id=resolved_project_id,
            params=embedding_params,
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize WatsonxEmbeddings ({resolved_model_id}): {str(e)}"
        )
