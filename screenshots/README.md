# Quest Analytics Assignment Screenshots Guide (60 Marks)

This directory stores the **six required evaluation screenshots** (10 marks each) for the peer-graded AI assignment:  
**Quest Analytics – AI-Powered Research Document Question Answering Assistant**.

---

## Required Screenshots Summary

| Task | Marks | Filename | What to Show |
| :---: | :---: | :--- | :--- |
| **1** | 10 | `pdf_loader.png` | PyPDFLoader imports, loading code, `loader.load()`, and document metadata / page count |
| **2** | 10 | `code_splitter.png` | `RecursiveCharacterTextSplitter` configuration (`chunk_size=1000`, `overlap=200`), splitting call, and chunk count |
| **3** | 10 | `embedding.png` | Embedding model initialization (`watsonx_embedding` / local), dimension output, chunk vectorization |
| **4** | 10 | `vectordb.png` | Chroma vector database initialization, embedding assignment, document insertion, local persistence (`./chroma_db`) |
| **5** | 10 | `retriever.png` | Semantic similarity retriever configuration (`as_retriever(search_type='similarity', k=4)`) and `retriever.invoke()` query execution |
| **6** | 10 | `QA_bot.png` | Running Gradio interface with Quest Analytics title, uploaded LoRA research PDF, question *"What is this paper talking about?"*, AI answer, and sources |

---

## Windows Screenshot Capture Instructions

Use the Windows Snipping Tool shortcut:
👉 **`Windows Key + Shift + S`**  
Then select the window or rectangular region, click the notification, and save the image into `C:\Users\SINDHU SURAJ\source\repos\RAG-based PDF Question Answering Bot\screenshots\<filename>.png`.

---

### Task 1: `screenshots/pdf_loader.png` (10 Marks)
- **Source File:** `utils/document_loader.py`
- **What to Capture:**
  1. Top imports: `from langchain_community.document_loaders import PyPDFLoader, TextLoader`
  2. The `document_loader()` function showing PDF validation and `loader.load()`.
  3. Terminal output from running `python tests/verify_assignment_pipeline.py` showing `[TASK 1: DOCUMENT LOADING]` with page count (11 pages) and sample metadata.

---

### Task 2: `screenshots/code_splitter.png` (10 Marks)
- **Source File:** `utils/text_splitter.py`
- **What to Capture:**
  1. `from langchain_text_splitters import RecursiveCharacterTextSplitter`
  2. The `text_splitter()` function displaying `chunk_size=1000`, `chunk_overlap=200`, and `length_function=len`.
  3. The `splitter.split_documents(documents)` call.
  4. Terminal verification output showing `38 chunks generated` and first chunk inspection.

---

### Task 3: `screenshots/embedding.png` (10 Marks)
- **Source File:** `utils/embeddings.py`
- **What to Capture:**
  1. `from langchain_ibm import WatsonxEmbeddings`
  2. The `watsonx_embedding()` function and `get_embeddings_model()`.
  3. The `validate_embedding_output()` function.
  4. Terminal output showing `ibm/slate-125m-english-rtrvr-v2`, embedding dimension `768`, and sample vector preview.

---

### Task 4: `screenshots/vectordb.png` (10 Marks)
- **Source File:** `utils/vector_store.py`
- **What to Capture:**
  1. `from langchain_chroma import Chroma`
  2. The `vector_database()` function showing `Chroma(...)`, `embedding_function`, `persist_directory="./chroma_db"`, and `vectordb.add_documents(...)`.
  3. Terminal output showing collection initialization and 38 chunks stored with deduplication.

---

### Task 5: `screenshots/retriever.png` (10 Marks)
- **Source File:** `utils/retriever.py`
- **What to Capture:**
  1. The `retriever_pipeline()` and `as_retriever()` call with `search_type="similarity"` and `search_kwargs={"k": 4}`.
  2. The retriever invocation: `retriever.invoke(query)`.
  3. Terminal output showing test query execution and the 4 retrieved chunks with source and page citations.

---

### Task 6: `screenshots/QA_bot.png` (10 Marks)
- **Running URL:** `http://127.0.0.1:7860`
- **What to Capture:**
  1. **Application Title:** `Quest Analytics | Research Document AI Assistant`
  2. **Uploaded PDF:** `A-Comprehensive-Review-of-Low-Rank-Adaptation-in-Large-Language-Models-for-Efficient-Parameter-Tuning-1.pdf`
  3. **Status Message:** `✅ Document processed successfully (11 pages, 38 chunks indexed into ChromaDB)`
  4. **Question Input:** `What is this paper talking about?`
  5. **AI-Generated Answer:** Factual synthesized response explaining LoRA and parameter efficiency.
  6. **Retrieved Sources:** Formatted citations showing document name and page numbers.

---

## Automated Verification Script

To produce all console outputs for Tasks 1–5 and test live question answering:
```powershell
.\.venv\Scripts\Activate.ps1
python tests/verify_assignment_pipeline.py
```
