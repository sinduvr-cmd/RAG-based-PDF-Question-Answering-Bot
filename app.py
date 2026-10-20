"""
Quest Analytics | Research Document AI Assistant
A professional Gradio web application leveraging LangChain, IBM watsonx.ai, ChromaDB,
and Sentence Transformers for question answering over uploaded scientific research documents.
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
from utils.embeddings import watsonx_embedding, get_embeddings_model
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
            "Please configure your .env file or enable Offline Local Mode for local testing."
        )
        return False, msg
    return True, "✅ IBM watsonx.ai credentials loaded successfully."


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
        return "❌ Please upload a valid PDF document.", {}

    file_path = file.name if hasattr(file, "name") else str(file)
    filename = Path(file_path).name

    try:
        # Step 1: Load PDF using PyPDFLoader
        documents = document_loader(file_path)
        num_pages = len(documents)

        # Step 2: Split text into chunks using RecursiveCharacterTextSplitter
        chunks = text_splitter(documents)
        num_chunks = len(chunks)

        # Step 3: Embeddings & Vector Store
        if use_mock_mode:
            from utils.local_retrieval import LocalRetriever
            retriever = LocalRetriever(chunks)
            status_prefix = "[Offline Local Mode Active] "
        else:
            has_creds, cred_msg = check_watsonx_credentials()
            if not has_creds:
                return (
                    f"Configuration Error: {cred_msg}\n\n"
                    "Tip: Check 'Enable Offline Local Mode' below to test with local embeddings.",
                    {},
                )
            # Resolve embeddings model (IBM watsonx or local Sentence Transformers fallback)
            embedding_fn, emb_name = get_embeddings_model()
            vectordb = vector_database(chunks, embedding_function=embedding_fn)
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            status_prefix = f"[{emb_name}] "

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
        return error_msg, {}


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
            result = question_answering(
                file=file_path,
                query=question,
                retriever=retriever,
                offline=True,
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
# Gradio User Interface Construction (Professional Blue / Dark Navy Theme)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    max-width: 1240px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    text-align: center;
    padding: 22px 28px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
    color: #f8fafc;
    border-radius: 12px;
    margin-bottom: 22px;
    box-shadow: 0 4px 10px -1px rgba(15, 23, 42, 0.4);
    border: 1px solid #1e40af;
}
.header-box h1 {
    font-size: 2.15rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.header-box p {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin: 0;
}
.status-box textarea {
    font-family: Consolas, Monaco, monospace !important;
    font-size: 0.92rem !important;
}
"""


def create_demo():
    """Build the UI when launching."""
    with gr.Blocks(title="Quest Analytics | Research Document AI Assistant") as demo:
        # Session state dictionary
        session_state = gr.State({})

        # Header section
        gr.HTML(
            """
            <div class="header-box">
                <h1>Quest Analytics | Research Document AI Assistant</h1>
                <p>Upload scientific research papers and ask questions using Retrieval-Augmented Generation.</p>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Upload & Document Processing Status
            with gr.Column(scale=5):
                gr.Markdown("### 📄 Document Ingestion")
                pdf_input = gr.File(
                    label="Upload PDF Research Document",
                    file_types=[".pdf", ".txt"],
                    file_count="single",
                )

                process_btn = gr.Button("⚙️ Process Document", variant="secondary")

                status_output = gr.Textbox(
                    label="Processing Status",
                    value="Upload a research PDF to begin.",
                    lines=4,
                    interactive=False,
                    elem_classes=["status-box"],
                )

                mock_checkbox = gr.Checkbox(
                    label="Enable Offline Local Mode (Source excerpts; no IBM generation)",
                    value=False,
                    info="Uncheck to use IBM watsonx foundation models configured in .env.",
                )

                with gr.Accordion("ℹ️ IBM watsonx Configuration Info", open=False):
                    creds_ok, creds_info = check_watsonx_credentials()
                    gr.Markdown(
                        f"**Connection Status:** {creds_info}\n\n"
                        f"• **Preferred LLM:** `mistralai/mixtral-8x7b-instruct-v01`\n"
                        f"• **Fallback LLM:** `ibm/granite-13b-chat-v2`\n"
                        f"• **Embedding Model:** `ibm/slate-125m-english-rtrvr-v2` (Local: `all-MiniLM-L6-v2`)\n"
                        f"• **Vector Store:** ChromaDB (`./chroma_db`)"
                    )

                clear_btn = gr.Button("🗑️ Clear / Reset Session", variant="secondary")

            # Right Column: Question Answering & Sources
            with gr.Column(scale=7):
                gr.Markdown("### 💬 Document Question Answering")
                question_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="e.g., What is this paper talking about? / What is Low-Rank Adaptation (LoRA)?",
                    lines=2,
                )

                ask_btn = gr.Button("🔍 Ask Question", variant="primary")

                answer_output = gr.Textbox(
                    label="AI-Generated Answer",
                    placeholder="The AI answer retrieved from the document will appear here...",
                    lines=6,
                    interactive=False,
                )

                sources_output = gr.Markdown(
                    label="Retrieved Document Sources",
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

        # Explicit Document Processing Button
        process_btn.click(
            fn=process_document,
            inputs=[pdf_input, mock_checkbox, session_state],
            outputs=[status_output, session_state],
        )

        # When Mock Mode checkbox is toggled, dynamically re-process the active document
        mock_checkbox.change(
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

    return demo


if __name__ == "__main__":
    print("Launching Quest Analytics | Research Document AI Assistant on http://127.0.0.1:7860 ...")
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
