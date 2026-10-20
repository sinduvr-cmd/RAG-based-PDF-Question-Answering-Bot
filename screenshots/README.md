# Screenshots Guide

This directory holds the required evaluation screenshot for the assignment: `QA_bot.png`.

## Required Screenshot: `QA_bot.png`

The screenshot must capture the running application interface with all of the following elements visible in a single frame:

1. **Application Title:** `PDF Research Assistant – RAG QA Bot`
2. **Uploaded PDF Filename:** (e.g., `sample_research_paper.pdf`)
3. **Document Processing Success Message:** `Document processed successfully.`
4. **User's Question:** (e.g., `What is the core architecture introduced in this research?`)
5. **AI-Generated Answer:** The answer retrieved from the document context.
6. **Source References:** The cited document name, page number, and text excerpt.

---

## How to Capture the Screenshot

### Option A: Using Real IBM watsonx.ai Credentials
1. Configure your `.env` file with your `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, and `WATSONX_URL`.
2. Start the application:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python app.py
   ```
3. Open your browser to `http://127.0.0.1:7860`.
4. Upload a research PDF (e.g., from `tests/fixtures/sample_research_paper.pdf`).
5. Confirm the status displays: `✅ Document processed successfully.`
6. Type your question in the textbox and click **Ask Question**.
7. Once the answer and sources appear, take a screenshot of the entire application window.
8. Save the file as:
   ```text
   screenshots/QA_bot.png
   ```

### Option B: Using the Automated Headless Browser or Mock Verification
When testing the layout before or after provisioning credentials, you can check the `Enable Offline Mock Mode` checkbox in the UI, upload the sample PDF, enter a question, and capture the interface.

---

## Checklist Before Submission

- [ ] Filename is exactly `QA_bot.png` in the `screenshots/` directory.
- [ ] Application title is clearly visible.
- [ ] Uploaded PDF filename is visible.
- [ ] Success message is visible.
- [ ] Question textbox contains the asked question.
- [ ] AI answer area contains the generated response.
- [ ] Source references show the file name and page citation.
