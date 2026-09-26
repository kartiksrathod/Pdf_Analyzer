# Contributing

Thanks for your interest in improving the PDF Intelligence (ChatPDF) application.

## Before you start

- The app must keep working end to end: **upload PDF(s) → `Process` → ask a question → answer**.
- Do not swap out the RAG stack (embedding model, vector store, LLM provider) inside a bug-fix PR.
  Open an issue or a clearly scoped PR first.
- Keep changes small and focused, and state what you verified.

## Development setup

```bash
git clone https://github.com/kartiksrathod/Pdf_Analyzer.git
cd Pdf_Analyzer/pdf_chat

python -m venv .venv
.venv\Scripts\activate                 # Windows
source .venv/bin/activate              # macOS / Linux

python -m pip install -r requirements.txt
ollama pull llama3.2                   # the model the app uses
streamlit run app.py
```

## Where things live

| File | Purpose |
|---|---|
| `app.py` | PDF ingestion helpers, RAG chain construction, Streamlit UI |
| `htmlTemplates.py` | chat bubble CSS and the user/assistant HTML templates |
| `requirements.txt` | runtime dependencies (keep the pins in sync with the tested stack) |
| `README.md` | setup, usage, limitations and troubleshooting |

## Guidelines

- Python style: PEP 8, 4-space indentation, f-strings; add type hints where they add clarity.
- Keep comments and UI text in English.
- **Never** put user, PDF or model text into HTML without `html.escape()`. Only the static templates
  in `htmlTemplates.py` are trusted markup (`unsafe_allow_html=True`).
- One submission must cause exactly one chain invocation: keep the `st.form` +
  `st.form_submit_button` gate, and keep rendering (`render_chat_history()`) separate from asking
  (`ask_question()`), so a Streamlit rerun can never re-submit a question.
- Derive chat roles from message types (`HumanMessage` / `AIMessage`), never from list positions.
- Report failures in the UI with `st.error` / `st.warning` instead of letting exceptions escape.
- If you add a dependency, pin the version you actually tested and confirm that
  `python -m pip install --dry-run -r requirements.txt` succeeds.
- Never commit secrets, `.env` files or anything from `.streamlit/secrets.toml`.

## Manual test checklist

Run through this before opening a pull request (with Ollama running and `llama3.2` pulled):

1. `streamlit run app.py` starts without errors and serves `http://localhost:8501`.
2. Upload one PDF and click `Process` — a chunk count is reported and the file is listed as indexed.
3. Ask a question — exactly one answer appears, grounded in your document.
4. Interact with another widget or reload the page — the transcript stays visible and **no** new
   answer is generated.
5. Ask a follow-up that refers to the previous turn — it is answered using the earlier turn.
6. Upload a different PDF and click `Process` again — the transcript clears and the new document is
   used.
7. Click `Process` with nothing selected, and again with a corrupt/scanned PDF — clear messages, no
   crash, and no stale document left active.
8. Ask a question before processing anything — a warning appears and no LLM call is made.

## Reporting bugs

Please include what you did, what you expected, what actually happened, the full error text, your OS,
your Python version (`python --version`) and your Ollama version/model (`ollama list`).

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
