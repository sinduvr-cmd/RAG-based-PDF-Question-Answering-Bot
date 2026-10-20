"""
Large Language Model (LLM) Configuration Module for PDF Research Assistant.
Configures and initializes IBM watsonx.ai instruction-following LLMs.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx

# Load environment variables
load_dotenv()

# Standard IBM watsonx.ai instruction-following models:
# - 'ibm/granite-13b-chat-v2' (Enterprise chat & instruction)
# - 'ibm/granite-3-8b-instruct' (Granite 3.0 compact instruction)
# - 'meta-llama/llama-3-3-70b-instruct' (High capability Llama 3.3)
# - 'meta-llama/llama-3-8b-instruct' (Llama 3 instruction)
DEFAULT_WATSONX_MODEL_ID = "ibm/granite-13b-chat-v2"


def get_llm(
    model_id: Optional[str] = None,
    apikey: Optional[str] = None,
    project_id: Optional[str] = None,
    url: Optional[str] = None,
    generation_params: Optional[Dict[str, Any]] = None,
) -> ChatWatsonx:
    """
    Initializes and returns an IBM watsonx.ai instruction-following LLM.

    Credentials and model settings are loaded from environment variables:
        WATSONX_APIKEY
        WATSONX_PROJECT_ID
        WATSONX_URL
        WATSONX_LLM_MODEL_ID (optional override)

    Parameters:
        model_id (Optional[str]): Watsonx model ID (defaults to WATSONX_LLM_MODEL_ID or granite-13b-chat-v2).
        apikey (Optional[str]): Explicit API key.
        project_id (Optional[str]): Explicit Project ID.
        url (Optional[str]): Explicit Watsonx URL.
        generation_params (Optional[Dict[str, Any]]): Text generation parameters.

    Returns:
        ChatWatsonx: Configured LangChain LLM instance.

    Raises:
        ValueError: If required credentials are missing.
        RuntimeError: If authentication or model initialization fails.
    """
    # 1. Resolve credentials
    resolved_apikey = apikey or os.getenv("WATSONX_APIKEY")
    resolved_project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
    resolved_url = url or os.getenv("WATSONX_URL")
    resolved_model_id = (
        model_id
        or os.getenv("WATSONX_LLM_MODEL_ID")
        or os.getenv("WATSONX_MODEL_ID")
        or DEFAULT_WATSONX_MODEL_ID
    )

    # 2. Validate credentials
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
            "Please configure them in your .env file or environment variables.\n"
            "Refer to .env.example for details."
        )

    # 3. Configure factual generation parameters (greedy / low temperature)
    default_params = {
        "max_tokens": 1200,
        "temperature": 0,
    }

    if generation_params:
        default_params.update(generation_params)

    # 4. Initialize ChatWatsonx
    try:
        llm = ChatWatsonx(
            model_id=resolved_model_id,
            url=resolved_url,
            apikey=resolved_apikey,
            project_id=resolved_project_id,
            params=default_params,
        )
        return llm
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChatWatsonx ({resolved_model_id}): {str(e)}"
        )
