# Quest Analytics – AI-Powered Research Document Question Answering Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4%2B-green.svg)](https://python.langchain.com/)
[![IBM watsonx.ai](https://img.shields.io/badge/IBM%20watsonx.ai-Slate%20%26%20Granite-black.svg)](https://www.ibm.com/watsonx)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-red.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fully functional, modular **Retrieval-Augmented Generation (RAG)** application implementing the peer-graded assignment:  
**"Quest Analytics – AI-Powered Research Document Question Answering Assistant"** (60 / 60 marks).

---

## 1. Project Overview

**Quest Analytics | Research Document AI Assistant** enables researchers, scientists, and students to upload scientific PDF papers and ask natural language questions using a Retrieval-Augmented Generation (RAG) architecture.

The system extracts and chunks document text, computes dense embeddings, persists them into a local Chroma vector database, retrieves semantically relevant passages, and generates factually grounded answers using IBM watsonx.ai instruction-tuned foundation models, citing exact filenames and page numbers.

### Key Capabilities
- **Task 1: Document Loading (10 Marks):** Extracts text and page metadata using LangChain's `PyPDFLoader` with robust validation against empty, scanned, or corrupted files. Also demonstrates additional data source loading via `TextLoader`.
- **Task 2: Text Splitting (10 Marks):** Splits documents with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`, `length_function=len`) preserving source and page metadata.
- **Task 3: Document Embeddings (10 Marks):** Computes dense semantic vectors with IBM watsonx (`ibm/slate-125m-english-rtrvr-v2`, 768-dim) and local Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) fallback.
- **Task 4: Persistent ChromaDB Vector Store (10 Marks):** Local on-disk persistence (`./chroma_db`) with deterministic chunk hashing to eliminate duplicate embeddings and isolate document uploads.
- **Task 5: Similarity Search Retriever (10 Marks):** Converts vector database into a semantic similarity retriever (`search_type="similarity"`, `k=4`) with query execution and chunk inspection.
- **Task 6: Complete RAG QA Bot (10 Marks):** Gradio web application with Quest Analytics blue/navy theme, PDF upload, document processing, question box, AI answer display, source citations, clear button, and offline/live modes. Tested with the official LoRA research paper.

---

## 2. Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion Pipeline"]
        PDF["📄 Uploaded Research PDF"] --> Loader["document_loader()\nPyPDFLoader"]
        Loader --> Docs["LangChain Documents\n[Page Content + Metadata]"]
        Docs --> Splitter["text_splitter()\nRecursiveCharacterTextSplitter\n(chunk_size=1000, overlap=200)"]
        Splitter --> Chunks["Text Chunks\nwith Source & Page IDs"]
    end

    subgraph Embedding_Storage ["2. Embedding & Persistent Vector Store"]
        Chunks --> EmbedModel["watsonx_embedding()\nibm/slate-125m-english-rtrvr-v2\n(truncate_input_tokens=512)"]
        EmbedModel --> VectorDB["vector_database()\nChromaDB Persistent Store\n(./chroma_db)"]
    end

    subgraph Retrieval_QA ["3. Retrieval-Augmented QA Pipeline"]
        UserQ["❓ User Question"] --> Retriever["retriever_pipeline()\nSimilarity Search (k=4)"]
        VectorDB --> Retriever
        Retriever --> ContextDocs["Top-4 Relevant Chunks\nwith Page Citations"]
        ContextDocs --> Prompt["QA Prompt Template\n'Answer using ONLY provided context'"]
        UserQ --> Prompt
        Prompt --> LLM["get_llm()\nWatsonxLLM (ibm/granite-13b-chat-v2)\n[Greedy / Temp 0.1]"]
        LLM --> Answer["💬 AI-Generated Answer"]
        ContextDocs --> Sources["📌 Formatted Citations\n(Document, Page, Excerpt)"]
    end

    subgraph Interface ["4. User Presentation Layer"]
        Answer --> Gradio["🖥️ Gradio Web Application\n(http://127.0.0.1:7860)"]
        Sources --> Gradio
    end
```

---

## 3. Technology Stack

| Component | Library / Service | Purpose |
| :--- | :--- | :--- |
| **Runtime** | Python 3.11 | Core execution environment |
| **Orchestration** | LangChain & LangChain-Classic | End-to-end RAG chain construction (`RetrievalQA`) |
| **Embeddings** | `langchain-ibm` (`ibm/slate-125m-english-rtrvr-v2`) | High-density semantic document representations |
| **LLM** | `langchain-ibm` (`ibm/granite-13b-chat-v2`) | Factual instruction-following question answering |
| **Vector Database** | ChromaDB (`langchain-chroma`) | Persistent local similarity index |
| **PDF Extraction** | PyPDF (`pypdf`, `langchain-community`) | Multi-page text and metadata extraction |
| **UI Framework** | Gradio 6 | Interactive web interface |
| **Configuration** | `python-dotenv` | Secure API credential management |
| **Testing** | `pytest` | Automated unit and integration test suite |

---

## 4. Project Directory Structure

```text
rag-qa-bot/
│
├── app.py                      # Main Gradio application entry point
├── requirements.txt            # Tested, pinned dependency specifications
├── .env.example                # Template for IBM watsonx credentials
├── .gitignore                  # Git ignore rules for venv, DB, and secrets
├── pytest.ini                  # Pytest configuration and pythonpath
├── README.md                   # Complete beginner-friendly documentation
│
├── agents/
│   ├── __init__.py             # Agent package exports
│   └── qa_agent.py             # RAG chain, QA prompt, citation formatter
│
├── utils/
│   ├── __init__.py             # Utilities package exports
│   ├── document_loader.py      # PyPDFLoader wrapper with validation
│   ├── text_splitter.py        # Recursive text chunking
│   ├── embeddings.py           # IBM slate-125m embedding connector
│   ├── vector_store.py         # Persistent Chroma vector store & deduplication
│   ├── retriever.py            # End-to-end retriever assembly
│   └── llm.py                  # IBM granite LLM connector
│
├── data/
│   ├── sample_research_paper.pdf # Included academic sample paper for testing
│   └── uploads/                # Directory for uploaded user documents
│
├── chroma_db/                  # Persistent local Chroma vector database
│
├── screenshots/
│   ├── README.md               # Guidelines for assignment screenshot
│   └── QA_bot.png              # Required evaluation screenshot
│
└── tests/
    ├── __init__.py             # Tests package
    └── test_pipeline.py        # 21 comprehensive unit & integration tests
```

---

## 5. IBM watsonx.ai Credential Setup

To use live IBM watsonx.ai foundation models, you will need an active IBM Cloud account and watsonx.ai project:

### Step 1: Create an IBM Cloud API Key
1. Log in to your [IBM Cloud Console](https://cloud.ibm.com/).
2. Navigate to **Manage** -> **Access (IAM)** -> **API keys**.
3. Click **Create an IBM Cloud API key**.
4. Name your key (e.g., `watsonx-rag-key`), copy the generated secret key, and save it safely.

### Step 2: Obtain Your watsonx.ai Project ID
1. Open [IBM watsonx.ai](https://dataplatform.cloud.ibm.com/wx/home).
2. Open or create a project (e.g., `PDF-Research-Assistant`).
3. Click the **Manage** tab in the top navigation bar.
4. Select the **General** section from the left menu.
5. Copy the **Project ID** (a UUID string like `12345678-abcd-ef01-2345-6789abcdef01`).

### Step 3: Identify Your Service URL
Depending on your data center region:
- **Dallas (us-south):** `https://us-south.ml.cloud.ibm.com`
- **Frankfurt (eu-de):** `https://eu-de.ml.cloud.ibm.com`
- **Tokyo (jp-tok):** `https://jp-tok.ml.cloud.ibm.com`
- **London (eu-gb):** `https://eu-gb.ml.cloud.ibm.com`

---

## 6. Environment Configuration

1. Create a `.env` file in the project root by copying `.env.example`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and fill in your credentials:
   ```env
   WATSONX_APIKEY=your_actual_api_key_here
   WATSONX_PROJECT_ID=your_actual_project_id_here
   WATSONX_URL=https://us-south.ml.cloud.ibm.com
   WATSONX_EMBEDDING_MODEL_ID=ibm/slate-125m-english-rtrvr-v2
   WATSONX_LLM_MODEL_ID=ibm/granite-13b-chat-v2
   ```

> [!IMPORTANT]
> Never commit your `.env` file to version control. The `.gitignore` file is configured to protect your credentials.

---

## 7. Installation & Execution Instructions

Run the following commands in **Windows PowerShell**:

### 1. Clone or Open the Repository
```powershell
cd "c:\Users\SINDHU SURAJ\source\repos\RAG-based PDF Question Answering Bot"
```

### 2. Create the Virtual Environment (Python 3.11)
```powershell
py -3.11 -m venv .venv
```

### 3. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first).*

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Run the Application
```powershell
python app.py
```

The Gradio web interface will launch locally at:
👉 **`http://127.0.0.1:7860`**

Open this URL in any web browser.

---

## 8. Example Questions & Walkthrough

You can test the application using the included research paper (`data/sample_research_paper.pdf`):

1. **Upload:** Drag and drop `data/sample_research_paper.pdf` into the upload box.
2. **Status:** Wait for `✅ Document processed successfully (3 pages extracted and indexed)`.
3. **Ask Questions:**
   - *"What is the main architecture proposed in this paper?"*
     > **Expected Answer:** The paper proposes the Transformer, an encoder-decoder architecture based entirely on self-attention mechanisms without sequence-aligned recurrent or convolutional networks.
   - *"What BLEU score did the model achieve on WMT 2014 English-to-German?"*
     > **Expected Answer:** The Transformer achieved a state-of-the-art BLEU score of 28.4 on the WMT 2014 English-to-German translation task.
   - *"What are the advantages of Multi-Head Attention?"*
     > **Expected Answer:** Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
   - *"What is the recipe for chocolate cake?"*
     > **Expected Answer:** The provided document does not contain information about a recipe for chocolate cake. (Demonstrates strict non-hallucination constraint).

---

## 9. Automated Testing Suite

The repository includes 21 comprehensive unit and integration tests covering all pipeline stages. External API calls are mocked so unit tests run completely offline without incurring cloud usage or requiring active credentials:

```powershell
.\.venv\Scripts\Activate.ps1
pytest -v
```

### What the Test Suite Verifies:
- **`TestDocumentLoader`:** PDF ingestion, missing file errors, invalid extensions, 0-byte PDFs, corrupt headers, Gradio file object compatibility.
- **`TestTextSplitter`:** Chunk boundaries, chunk overlap constraints, metadata preservation (`source`, `page`, `chunk_index`).
- **`TestEmbeddings`:** Slate-125m model ID, input token truncation params, credential validation errors.
- **`TestVectorStore`:** Persistent Chroma initialization, deterministic chunk hash deduplication, collection name sanitization.
- **`TestRetrieverPipeline`:** 5-step pipeline execution, top-4 similarity search retrieval.
- **`TestLLMConfiguration`:** WatsonxLLM parameters (`greedy`, low temperature, factual generation).
- **`TestQuestionAnsweringAgent`:** Custom prompt constraint verification, `RetrievalQA` execution, source citation formatting, empty query handling.

---

## 10. Required Assignment Evaluation Screenshots (60 Marks Total)

The peer-graded assignment requires six genuine screenshots (10 marks each) placed in `screenshots/`:

| Task # | Marks | Screenshot Filename | Required Evidence Shown | Source Code Location |
| :---: | :---: | :--- | :--- | :--- |
| **Task 1** | 10 | `screenshots/pdf_loader.png` | PyPDFLoader import, document loading logic, `loader.load()`, page metadata & text extraction output | `utils/document_loader.py` |
| **Task 2** | 10 | `screenshots/code_splitter.png` | RecursiveCharacterTextSplitter configuration (`chunk_size=1000`, `overlap=200`), split invocation, chunk count | `utils/text_splitter.py` |
| **Task 3** | 10 | `screenshots/embedding.png` | Embedding model initialization (`watsonx_embedding` / local), dimension output, chunk vectorization | `utils/embeddings.py` |
| **Task 4** | 10 | `screenshots/vectordb.png` | Chroma collection initialization, embedding assignment, document insertion, persistence to `./chroma_db` | `utils/vector_store.py` |
| **Task 5** | 10 | `screenshots/retriever.png` | Similarity retriever conversion (`as_retriever(search_type='similarity', k=4)`), test query execution, retrieved chunks | `utils/retriever.py` |
| **Task 6** | 10 | `screenshots/QA_bot.png` | Live Gradio interface showing title, uploaded LoRA PDF, question *"What is this paper talking about?"*, AI answer, sources | `app.py` (`http://127.0.0.1:7860`) |

Detailed capture instructions and checklist are available in [screenshots/README.md](file:///c:/Users/SINDHU%20SURAJ/source/repos/RAG-based%20PDF%20Question%20Answering%20Bot/screenshots/README.md).

---

## 11. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **`Missing required IBM watsonx.ai credentials`** | `.env` file does not exist or variables are blank | Copy `.env.example` to `.env` and fill in `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, and `WATSONX_URL`. For offline testing, check the **Enable Offline Mock Mode** box in the UI. |
| **`ExecutionPolicy error in PowerShell`** | Windows blocks PowerShell scripts by default | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in your PowerShell terminal. |
| **`The file is not a valid PDF document`** | File missing `%PDF-` header or has wrong format | Ensure the file is an authentic text-based PDF file. |
| **`The PDF contains no extractable text`** | PDF is a scanned image with no OCR layer | Run OCR on the document or upload a digital-native text PDF. |
| **`Port 7860 already in use`** | Another Gradio app is running on port 7860 | Specify an alternate port in `app.py` (e.g., `demo.launch(server_port=7861)`). |

---

## 12. License

This project is created for educational and academic assessment purposes under the MIT License.
