"""
Large Language Model (LLM) Configuration Module for Quest Analytics Assistant.
Configures and initializes IBM watsonx.ai instruction-following foundation models.
Prefers 'mistralai/mixtral-8x7b-instruct-v01' as permitted by the assignment rubric;
falls back to 'ibm/granite-13b-chat-v2' if Mixtral is unprovisioned in the account region.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx, WatsonxLLM

# Load environment variables
load_dotenv()

PREFERRED_WATSONX_MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"
FALLBACK_WATSONX_MODEL_ID = "ibm/granite-13b-chat-v2"


def get_llm(
    model_id: Optional[str] = None,
    apikey: Optional[str] = None,
    project_id: Optional[str] = None,
    url: Optional[str] = None,
    generation_params: Optional[Dict[str, Any]] = None,
):
    """
    Initializes and returns an IBM watsonx.ai instruction-following LLM.
    Prefers 'mistralai/mixtral-8x7b-instruct-v01' if available.

    Credentials and settings are loaded from .env:
        WATSONX_APIKEY
        WATSONX_PROJECT_ID
        WATSONX_URL
        WATSONX_LLM_MODEL_ID
    """

    # 1. Resolve credentials
    resolved_apikey = apikey or os.getenv("WATSONX_APIKEY")
    resolved_project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
    resolved_url = url or os.getenv("WATSONX_URL")
    resolved_model_id = (
        model_id
        or os.getenv("WATSONX_LLM_MODEL_ID")
        or PREFERRED_WATSONX_MODEL_ID
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
            "Please configure them in your .env file."
        )

    # 3. Configure generation parameters (greedy/low temperature for factual answers)
    default_params = {
        "max_tokens": 1200,
        "temperature": 0.0,
    }

    if generation_params:
        default_params.update(generation_params)

    # 4. Attempt initialization with preferred model, with fallback to granite if requested model is unavailable
    try:
        llm = ChatWatsonx(
            model_id=resolved_model_id,
            url=resolved_url,
            apikey=resolved_apikey,
            project_id=resolved_project_id,
            params=default_params,
        )
        return llm
    except Exception as primary_err:
        if resolved_model_id != FALLBACK_WATSONX_MODEL_ID:
            try:
                # Attempt fallback to IBM Granite
                llm = ChatWatsonx(
                    model_id=FALLBACK_WATSONX_MODEL_ID,
                    url=resolved_url,
                    apikey=resolved_apikey,
                    project_id=resolved_project_id,
                    params=default_params,
                )
                return llm
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to initialize ChatWatsonx ({resolved_model_id}): {str(primary_err)}"
        )
