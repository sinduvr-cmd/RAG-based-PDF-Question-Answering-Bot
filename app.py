"""
PDF Research Assistant – RAG QA Bot
A Gradio-based web application leveraging LangChain, IBM watsonx.ai, and ChromaDB
for question answering over uploaded PDF research documents.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import gradio as gr
from dotenv import load_dotenv

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

from utils.document_loader import document_loader
from utils.text_splitter import text_splitter
from utils.embeddings import watsonx_embedding
from utils.vector_store import vector_database
from utils.retriever import retriever_pipeline
from utils.llm import get_llm
from agents.qa_agent import question_answering, format_sources, clear_retriever_cache


# ---------------------------------------------------------------------------
# Helper: Check IBM watsonx Credentials
# ---------------------------------------------------------------------------
def check_watsonx_credentials() -> Tuple[bool, str]:
    """Checks whether the necessary IBM watsonx credentials exist in environment."""
    apikey = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")

    missing = []
    if not apikey:
        missing.append("WATSONX_APIKEY")
    if not project_id:
        missing.append("WATSONX_PROJECT_ID")
    if not url:
        missing.append("WATSONX_URL")

    if missing:
        msg = (
            f"⚠️ Watsonx Credentials Incomplete: Missing {', '.join(missing)} in .env. "
            "Please configure your .env file or enable Mock Mode for offline UI testing."
        )
        return False, msg
    return True, "✅ IBM watsonx.ai credentials loaded successfully."


from langchain_core.language_models.fake import FakeListLLM

class OfflineMockLLM(FakeListLLM):
    """Mock LLM used strictly when mock mode is enabled for testing/screenshots."""

    def __init__(self):
        default_answer = (
            "Based on the provided document, the Transformer architecture is based entirely "
            "on self-attention mechanisms without relying on sequence-aligned recurrent or "
            "convolutional neural networks. It introduces Multi-Head Attention to attend to "
            "information from different representation subspaces at different positions."
        )
        super().__init__(responses=[default_answer])


# ---------------------------------------------------------------------------
# Core Event Handlers
# ---------------------------------------------------------------------------
def process_document(
    file,
    use_mock_mode: bool,
    session_state: dict,
) -> Tuple[str, dict]:
    """
    Handles PDF upload, parses document, generates chunks, and builds the retriever.
    """
    if file is None:
        return "❌ Please upload a valid PDF document.", session_state

    file_path = file.name if hasattr(file, "name") else str(file)
    filename = Path(file_path).name

    try:
        # Step 1: Load PDF
        documents = document_loader(file_path)
        num_pages = len(documents)

        # Step 2: Split text into chunks
        chunks = text_splitter(documents)
        num_chunks = len(chunks)

        # Step 3: Embeddings & Vector Store
        if use_mock_mode:
            from langchain_core.embeddings import FakeEmbeddings
            embedding_fn = FakeEmbeddings(size=128)
            status_prefix = "[Mock Mode Active] "
        else:
            has_creds, cred_msg = check_watsonx_credentials()
            if not has_creds:
                return (
                    f"❌ Configuration Error: {cred_msg}\n"
                    "Tip: Check 'Enable Offline Mock Mode' below if you want to test the UI without IBM credentials.",
                    session_state,
                )
            embedding_fn = watsonx_embedding()
            status_prefix = ""

        # Step 4: Build Vector Database & Retriever
        col_name = f"mock_{Path(file_path).stem}" if use_mock_mode else None
        vectordb = vector_database(
            chunks,
            embedding_function=embedding_fn,
            collection_name=col_name,
        )
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Update session state
        session_state = {
            "file_path": file_path,
            "filename": filename,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "retriever": retriever,
            "use_mock_mode": use_mock_mode,
        }

        success_msg = (
            f"✅ {status_prefix}Document processed successfully.\n"
            f"• File: {filename}\n"
            f"• Extracted Pages: {num_pages}\n"
            f"• Indexed Chunks: {num_chunks}"
        )
        return success_msg, session_state

    except Exception as e:
        error_msg = f"❌ Processing Error: {str(e)}"
        return error_msg, session_state


def answer_question(
    question: str,
    session_state: dict,
) -> Tuple[str, str]:
    """
    Executes the RAG pipeline to answer the user's question using the active document retriever.
    """
    if not question or not question.strip():
        return "Please enter a question about the uploaded document.", ""

    if not session_state or "retriever" not in session_state:
        return (
            "Please upload and process a PDF document before asking questions.",
            "No source references available.",
        )

    retriever = session_state.get("retriever")
    use_mock_mode = session_state.get("use_mock_mode", False)
    file_path = session_state.get("file_path")

    try:
        if use_mock_mode:
            # Use offline mock LLM
            mock_llm = OfflineMockLLM()
            result = question_answering(
                file=file_path,
                query=question,
                retriever=retriever,
                llm=mock_llm,
            )
        else:
            has_creds, cred_msg = check_watsonx_credentials()
            if not has_creds:
                return f"❌ Configuration Error: {cred_msg}", ""
            result = question_answering(
                file=file_path,
                query=question,
                retriever=retriever,
            )

        return result["answer"], result["sources"]

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}", "No sources available due to error."


def clear_session() -> Tuple[None, str, str, str, str, dict]:
    """Clears inputs, outputs, cache, and resets session state."""
    clear_retriever_cache()
    return None, "Upload a research PDF to begin.", "", "", "", {}


# ---------------------------------------------------------------------------
# Gradio User Interface Construction
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    text-align: center;
    padding: 18px 24px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #f8fafc;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.header-box h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #ffffff;
}
.header-box p {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
}
.status-box textarea {
    font-family: monospace !important;
    font-size: 0.92rem !important;
}
"""

with gr.Blocks(title="PDF Research Assistant – RAG QA Bot") as demo:
    # Session state dictionary
    session_state = gr.State({})

    # Header section
    gr.HTML(
        """
        <div class="header-box">
            <h1>PDF Research Assistant – RAG QA Bot</h1>
            <p>Upload your research PDF and ask questions using AI-powered document retrieval.</p>
        </div>
        """
    )

    with gr.Row():
        # Left Column: Upload & Document Processing Status
        with gr.Column(scale=5):
            gr.Markdown("### 📄 Document Ingestion")
            pdf_input = gr.File(
                label="Upload PDF Research Document",
                file_types=[".pdf"],
                file_count="single",
            )

            status_output = gr.Textbox(
                label="Document Processing Status",
                value="Upload a research PDF to begin.",
                lines=4,
                interactive=False,
                elem_classes=["status-box"],
            )

            mock_checkbox = gr.Checkbox(
                label="Enable Offline Mock Mode (For testing UI without live IBM credentials)",
                value=False,
                info="Uncheck to use live IBM watsonx.ai slate-125m embeddings and granite LLM.",
            )

            with gr.Accordion("ℹ️ IBM watsonx Configuration Info", open=False):
                creds_ok, creds_info = check_watsonx_credentials()
                gr.Markdown(
                    f"**Connection Status:** {creds_info}\n\n"
                    "• **Embedding Model:** `ibm/slate-125m-english-rtrvr-v2`\n"
                    "• **LLM Model:** `ibm/granite-13b-chat-v2`\n"
                    "• **Vector Store:** ChromaDB (`./chroma_db`)"
                )

            clear_btn = gr.Button("🗑️ Clear / Reset Session", variant="secondary")

        # Right Column: Question Answering & Sources
        with gr.Column(scale=7):
            gr.Markdown("### 💬 Document Question Answering")
            question_input = gr.Textbox(
                label="Ask a Question",
                placeholder="e.g., What are the main contributions and findings of this research?",
                lines=2,
            )

            ask_btn = gr.Button("🔍 Ask Question", variant="primary")

            answer_output = gr.Textbox(
                label="AI-Generated Answer",
                placeholder="The AI answer retrieved from the document will appear here...",
                lines=5,
                interactive=False,
            )

            sources_output = gr.Markdown(
                label="Source References",
                value="*Source citations and page references will appear here after asking a question.*",
            )

    # -----------------------------------------------------------------------
    # Wire Interactive Events
    # -----------------------------------------------------------------------
    # When PDF is uploaded or changed, trigger document processing
    pdf_input.change(
        fn=process_document,
        inputs=[pdf_input, mock_checkbox, session_state],
        outputs=[status_output, session_state],
    )

    # When Ask Question is clicked
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, session_state],
        outputs=[answer_output, sources_output],
    )

    # Allow pressing Enter in question box
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, session_state],
        outputs=[answer_output, sources_output],
    )

    # Clear button
    clear_btn.click(
        fn=clear_session,
        inputs=[],
        outputs=[pdf_input, status_output, question_input, answer_output, sources_output, session_state],
    )


if __name__ == "__main__":
    print("Launching PDF Research Assistant – RAG QA Bot on http://127.0.0.1:7860 ...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
