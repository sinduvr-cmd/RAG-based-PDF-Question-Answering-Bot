"""
Configuration Module for Quest Analytics Research Document AI Assistant.
Centralizes environment variables, model choices, RAG hyperparameters, and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# IBM watsonx.ai Credentials
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# LLM Model Configuration
# Preferred assignment model: mistralai/mixtral-8x7b-instruct-v01
# Fallback model: ibm/granite-13b-chat-v2
PREFERRED_LLM_MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"
FALLBACK_LLM_MODEL_ID = "ibm/granite-13b-chat-v2"
WATSONX_LLM_MODEL_ID = os.getenv("WATSONX_LLM_MODEL_ID") or PREFERRED_LLM_MODEL_ID

# Embedding Configuration
# Cloud: ibm/slate-125m-english-rtrvr-v2
# Local fallback: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
WATSONX_EMBEDDING_MODEL_ID = os.getenv("WATSONX_EMBEDDING_MODEL_ID", "ibm/slate-125m-english-rtrvr-v2")
LOCAL_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Hyperparameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4

# Storage Paths
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
UPLOADS_DIR = str(PROJECT_ROOT / "data" / "uploads")
SCREENSHOTS_DIR = str(PROJECT_ROOT / "screenshots")
