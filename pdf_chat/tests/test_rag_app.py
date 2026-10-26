"""Automated tests for the PDF Intelligence (ChatPDF) application.

Run from the pdf_chat directory:

    python -m pytest tests -v

Design
------
* Deterministic tests use a fake chat model, so they never need Ollama.
* Tests marked ``live`` use the real Ollama server and the real ``llama3.2``
  model; they are skipped automatically when Ollama (or the model) is missing.
* PDFs are generated in memory with PyPDF2, which is already a runtime
  dependency - no test-only PDF library is required.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import pytest
from pydantic import Field

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app as chatpdf  # noqa: E402
import langchain_ollama  # noqa: E402
import streamlit  # noqa: E402
from langchain_classic.chains import ConversationalRetrievalChain  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(APP_DIR / "app.py")
OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = chatpdf.LLM_MODEL

# Facts that exist in exactly one page of exactly one test document.
A_PAGE_1 = (
    "A database transaction is a unit of work that guarantees the ACID properties. "
    "Databases keep a write ahead log so that committed transactions can be recovered."
)
A_PAGE_2 = (
    "Database indexes are usually implemented as B plus trees, which keep range "
    "queries logarithmic in the number of stored records."
)
B_PAGE_1 = (
    "An operating system schedules threads with round robin scheduling and priority "
    "queues to share the processor fairly."
)
B_PAGE_2 = (
    "Virtual memory maps virtual pages to physical frames through a page table and "
    "supports demand paging on page faults."
)


def make_pdf(pages):
    """Build a PDF with extractable Helvetica text, written directly as bytes.

    The bytes are assembled by hand on purpose: PyPDF2 3.0.1's ``add_blank_page``
    does not persist later ``/Resources``/``/Contents`` mutations, so a writer
    built PDF would contain no text at all. A falsy page entry produces an empty
    page, which is how blank/scanned pages behave for text extraction.
    """
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [ {kids} ] >>")
    font_number = 3 + 2 * len(pages)

    for index, text in enumerate(pages):
        page_number = 3 + 2 * index
        stream_number = page_number + 1
        resources = f"<< /Font << /F1 {font_number} 0 R >> >>" if text else "<< >>"
        contents = f"{stream_number} 0 R" if text else "[]"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources {resources} /Contents {contents} >>"
        )
        lines = ["BT /F1 11 Tf 14 TL 72 720 Td"]
        if text:
            for line in [text[i:i + 90] for i in range(0, len(text), 90)]:
                safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
                lines.append(f"({safe}) Tj T*")
        lines.append("ET")
        body = "\n".join(lines) if text else "BT ET"
        objects.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")

    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")

    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{start}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


class FakeUpload(io.BytesIO):
    """Minimal stand-in for streamlit's UploadedFile."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


class RecordingChatModel(BaseChatModel):
    """Deterministic stand-in for llama3.2.

    * condense step -> echoes the follow-up question (and records the prompt)
    * answer step   -> answers strictly from the labelled retrieved context
    * records every prompt so tests can assert on the real prompt that was sent
    """

    prompts: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = "\n".join(str(getattr(message, "content", message)) for message in messages)
        self.prompts.append(prompt)

        if "Standalone question:" in prompt:
            follow_up = prompt.rsplit("Follow Up Input:", 1)[-1]
            answer = follow_up.split("Standalone question:")[0].strip()
        elif "Retrieved context:" in prompt:
            answer = self._answer_from_context(prompt)
        else:
            answer = "no context provided"

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    @staticmethod
    def _answer_from_context(prompt: str) -> str:
        context = prompt.split("Retrieved context:", 1)[1]
        blocks = [block.strip() for block in context.split("[Source: ") if block.strip()]
        if not blocks:
            return chatpdf.NOT_FOUND_MESSAGE
        lines = blocks[0].splitlines()
        # lines[0] is the "file.pdf, Page: n]" label, lines[1] is the chunk text
        return "Answer: " + (lines[1].strip() if len(lines) > 1 else chatpdf.NOT_FOUND_MESSAGE)


def prompt_for(prompts, marker):
    """Return the last recorded prompt containing ``marker``."""
    matches = [prompt for prompt in prompts if marker in prompt]
    assert matches, f"no recorded prompt contained {marker!r}"
    return matches[-1]


def ollama_status():
    """Return ``(running, model_available)`` for the local Ollama server."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=4) as response:
            models = json.load(response).get("models", [])
    except Exception:
        return False, False
    names = {model.get("name", "") for model in models}
    return True, any(name.split(":")[0] == LLM_MODEL for name in names)


OLLAMA_RUNNING, MODEL_AVAILABLE = ollama_status()
live = pytest.mark.skipif(
    not (OLLAMA_RUNNING and MODEL_AVAILABLE),
    reason="needs a running Ollama server with the llama3.2 model pulled",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def two_pdf_uploads():
    return [
        FakeUpload(make_pdf([A_PAGE_1, A_PAGE_2]), "A.pdf"),
        FakeUpload(make_pdf([B_PAGE_1, B_PAGE_2]), "B.pdf"),
    ]


@pytest.fixture(scope="module")
def indexed(two_pdf_uploads):
    """Full ingestion of A.pdf + B.pdf into a real FAISS store (embeddings run once)."""
    page_documents, report = chatpdf.extract_pdf_pages(two_pdf_uploads)
    chunks = chatpdf.build_chunks(page_documents)
    return {
        "pages": page_documents,
        "report": report,
        "chunks": chunks,
        "store": chatpdf.get_vectorstore(chunks),
    }


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace ChatOllama with a deterministic model (production code untouched)."""
    model = RecordingChatModel()
    monkeypatch.setattr(chatpdf, "ChatOllama", lambda *args, **kwargs: model)
    return model


# --------------------------------------------------------------------------- #
# Test 6 - metadata survives PDF -> page -> chunk -> FAISS -> retrieval
# --------------------------------------------------------------------------- #
def test_page_aware_extraction_records_pages(indexed):
    report = {entry["source"]: entry for entry in indexed["report"]}
    assert set(report) == {"A.pdf", "B.pdf"}
    for entry in report.values():
        assert entry["error"] is None
        assert entry["pages_indexed"] == 2
        assert entry["failed_pages"] == []
        assert entry["empty_pages"] == []

    page_pairs = {(page.metadata["source"], page.metadata["page"]) for page in indexed["pages"]}
    assert page_pairs == {("A.pdf", 1), ("A.pdf", 2), ("B.pdf", 1), ("B.pdf", 2)}


def test_chunks_keep_source_and_page(indexed):
    for chunk in indexed["chunks"]:
        metadata = chunk.metadata
        assert chunk.page_content.strip()
        assert metadata["source"] in {"A.pdf", "B.pdf"}
        assert metadata["page"] in {1, 2}
        assert metadata["document_id"] in {"doc1", "doc2"}
        assert metadata["chunk_id"].startswith(metadata["document_id"])
    # a chunk may never mix text from two pages
    for chunk in indexed["chunks"]:
        if chunk.metadata["source"] == "A.pdf" and chunk.metadata["page"] == 2:
            assert "B plus trees" in chunk.page_content
            assert "ACID" not in chunk.page_content


def test_metadata_survives_faiss_and_retrieval(indexed):
    assert indexed["store"].index.ntotal == len(indexed["chunks"])

    retriever = chatpdf.ScopedRetriever(vectorstore=indexed["store"], k=chatpdf.RETRIEVER_K)
    documents = retriever.invoke("What do database indexes use for range queries?")

    assert documents, "retriever returned nothing"
    for document in documents:
        assert {"source", "page", "document_id", "chunk_id", "score"} <= set(document.metadata)
        assert isinstance(document.metadata["page"], int)
    top = documents[0]
    assert (top.metadata["source"], top.metadata["page"]) == ("A.pdf", 2)
    assert "B plus trees" in top.page_content


# --------------------------------------------------------------------------- #
# Tests 1 + 2 - answered from the right document, with the right page, while
# multiple documents stay distinguishable
# --------------------------------------------------------------------------- #
def test_question_about_document_a_retrieves_document_a(indexed, fake_llm):
    chain = chatpdf.get_conversationchain(indexed["store"])
    response = chain.invoke({"question": "What do database indexes use?"})

    sources = chatpdf.collect_sources(response["source_documents"])
    assert {"source": "A.pdf", "page": 2} in sources
    assert sources[0]["source"] == "A.pdf"
    assert "B plus trees" in response["answer"]


def test_question_about_document_b_retrieves_document_b(indexed, fake_llm):
    chain = chatpdf.get_conversationchain(indexed["store"])
    response = chain.invoke({"question": "How does virtual memory map pages to frames?"})

    sources = chatpdf.collect_sources(response["source_documents"])
    assert sources[0]["source"] == "B.pdf"
    assert any(source["page"] == 2 for source in sources if source["source"] == "B.pdf")


# --------------------------------------------------------------------------- #
# Test 3 - the prompt grounds the model and forbids inventing content
# --------------------------------------------------------------------------- #
def test_answer_prompt_grounds_the_model_in_the_retrieved_context(indexed, fake_llm):
    chain = chatpdf.get_conversationchain(indexed["store"])
    chain.invoke({"question": "What do database indexes use?"})

    prompt = prompt_for(fake_llm.prompts, "Retrieved context:")
    assert chatpdf.NOT_FOUND_MESSAGE in prompt
    assert "Use ONLY the retrieved context" in prompt
    assert "[Source: A.pdf, Page: 2]" in prompt


# --------------------------------------------------------------------------- #
# Test 4 - follow-up questions keep their conversational context
# --------------------------------------------------------------------------- #
def test_follow_up_question_is_condensed_with_the_previous_turn(indexed, fake_llm):
    chain = chatpdf.get_conversationchain(indexed["store"])
    chain.invoke({"question": "What do database indexes use?"})
    chain.invoke({"question": "and why does that help range queries?"})

    condense_prompt = prompt_for(fake_llm.prompts, "Standalone question:")
    assert "What do database indexes use?" in condense_prompt, "chat history was not passed on"
    assert "and why does that help range queries?" in condense_prompt


# --------------------------------------------------------------------------- #
# Document filtering (item 5)
# --------------------------------------------------------------------------- #
def test_document_filter_returns_only_that_document(indexed):
    document_id = next(
        entry["document_id"] for entry in indexed["report"] if entry["source"] == "B.pdf"
    )
    retriever = chatpdf.ScopedRetriever(
        vectorstore=indexed["store"], k=chatpdf.RETRIEVER_K, document_id=document_id
    )
    documents = retriever.invoke("What do database indexes use?")  # about A.pdf
    assert documents, "scoped retrieval returned nothing"
    assert all(document.metadata["source"] == "B.pdf" for document in documents)


def test_unfiltered_retrieval_still_searches_every_document(indexed):
    retriever = chatpdf.ScopedRetriever(vectorstore=indexed["store"], k=chatpdf.RETRIEVER_K)
    documents = retriever.invoke("Compare database indexes with virtual memory")
    assert {document.metadata["source"] for document in documents} == {"A.pdf", "B.pdf"}


# --------------------------------------------------------------------------- #
# Test 7 - empty / unreadable pages must not break ingestion
# --------------------------------------------------------------------------- #
def test_blank_pages_are_reported_and_skipped():
    uploads = [FakeUpload(make_pdf([A_PAGE_1, "", A_PAGE_2]), "mixed.pdf")]
    pages, report = chatpdf.extract_pdf_pages(uploads)

    assert report[0]["empty_pages"] == [2]
    assert report[0]["failed_pages"] == []
    assert report[0]["pages_indexed"] == 2, "an empty page must never count as indexed"
    assert {page.metadata["page"] for page in pages} == {1, 3}

    chunks = chatpdf.build_chunks(pages)
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 3}


def test_unreadable_file_does_not_break_the_batch():
    uploads = [
        FakeUpload(b"this is not a pdf at all", "broken.pdf"),
        FakeUpload(make_pdf([B_PAGE_1]), "good.pdf"),
    ]
    pages, report = chatpdf.extract_pdf_pages(uploads)
    by_source = {entry["source"]: entry for entry in report}

    assert by_source["broken.pdf"]["error"] is not None
    assert by_source["broken.pdf"]["pages_indexed"] == 0
    assert by_source["good.pdf"]["pages_indexed"] == 1
    assert [page.metadata["source"] for page in pages] == ["good.pdf"]
    assert chatpdf.build_chunks(pages), "the readable file must still be indexed"


def test_same_filename_is_disambiguated():
    uploads = [
        FakeUpload(make_pdf([A_PAGE_1]), "notes.pdf"),
        FakeUpload(make_pdf([B_PAGE_1]), "notes.pdf"),
    ]
    pages, report = chatpdf.extract_pdf_pages(uploads)

    assert [entry["source"] for entry in report] == ["notes.pdf", "notes.pdf (2)"]
    assert len({entry["document_id"] for entry in report}) == 2
    assert {page.metadata["document_id"] for page in pages} == {"doc1", "doc2"}


def test_whitespace_only_page_produces_no_chunks():
    pages, report = chatpdf.extract_pdf_pages([FakeUpload(make_pdf(["   \n\n  "]), "spaces.pdf")])
    assert pages == [], "a whitespace-only page has no text to index"
    assert report[0]["empty_pages"] == [1]
    assert chatpdf.build_chunks(pages) == []


def test_chunk_configuration_is_unchanged():
    """chunk_size=1000 / chunk_overlap=200 must stay exactly as before."""
    assert (chatpdf.CHUNK_SIZE, chatpdf.CHUNK_OVERLAP) == (1000, 200)

    splitter = chatpdf.build_text_splitter()
    # langchain-text-splitters 1.x kept these fields private, older versions public
    configured_size = getattr(splitter, "chunk_size", None) or getattr(splitter, "_chunk_size", None)
    configured_overlap = (
        getattr(splitter, "chunk_overlap", None) or getattr(splitter, "_chunk_overlap", None)
    )
    assert (configured_size, configured_overlap) == (1000, 200)

    text = " ".join(f"word{index}" for index in range(400))
    chunks = splitter.split_text(text)
    assert len(chunks) > 1, "a 2.8k character page must be split"
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert set(chunks[0].split()) & set(chunks[1].split()), "consecutive chunks must overlap"

    assert chatpdf.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert chatpdf.LLM_MODEL == "llama3.2"
    assert chatpdf.RETRIEVER_K == 4


# --------------------------------------------------------------------------- #
# Citations (item 2): only real retrieved sources, de-duplicated, escaped
# --------------------------------------------------------------------------- #
def test_no_citation_is_invented():
    assert chatpdf.collect_sources([]) == []
    assert chatpdf.collect_sources(None) == []
    assert chatpdf.format_sources([]) == ""


def test_citations_are_deduplicated_and_keep_retrieval_order():
    documents = [
        Document(page_content="x", metadata={"source": "A.pdf", "page": 2}),
        Document(page_content="y", metadata={"source": "A.pdf", "page": 2}),
        Document(page_content="z", metadata={"source": "B.pdf", "page": 1}),
    ]
    assert chatpdf.collect_sources(documents) == [
        {"source": "A.pdf", "page": 2},
        {"source": "B.pdf", "page": 1},
    ]


def test_citations_are_escaped():
    documents = [Document(page_content="x", metadata={"source": "<script>a.pdf", "page": 3})]
    block = chatpdf.format_sources(chatpdf.collect_sources(documents))
    assert "<script>" not in block
    assert "&lt;script&gt;" in block
    assert "Page 3" in block


def test_document_without_metadata_is_not_turned_into_a_fake_citation():
    assert chatpdf.collect_sources([Document(page_content="x", metadata={})]) == [
        {"source": "unknown document", "page": None}
    ]


# --------------------------------------------------------------------------- #
# Streamlit UI level (Tests 1-5 through the real app) + Prompt 1 regression
# --------------------------------------------------------------------------- #
def start_app(monkeypatch, uploads, fake_llm=True):
    """Boot app.py with the file uploader (and Ollama) replaced."""
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: uploads["current"])
    if fake_llm:
        monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda *a, **k: RecordingChatModel())
    app = AppTest.from_file(APP_PATH, default_timeout=300)
    app.run()
    return app


def click(app, label):
    next(button for button in app.button if button.label == label).click()
    app.run()
    return app


def ask(app, question):
    app.text_input[0].set_value(question)
    return click(app, "Ask")


class ScriptedConversation:
    """Fake chain that returns HTML payloads, to check the escaping path."""

    def invoke(self, payload):
        return {
            "chat_history": [
                HumanMessage(content="<script>alert('xss')</script>"),
                AIMessage(content="<img src=x onerror=alert(1)>"),
            ],
            "answer": "<img src=x onerror=alert(1)>",
        }


def test_app_boots_without_exceptions(monkeypatch, two_pdf_uploads):
    app = start_app(monkeypatch, {"current": two_pdf_uploads})
    assert not app.exception, [e.value for e in app.exception]
    assert any(button.label == "Ask" for button in app.button)


def test_one_submission_runs_the_chain_once_and_reruns_do_not_resubmit(monkeypatch, two_pdf_uploads):
    calls = {"count": 0}
    original = ConversationalRetrievalChain.invoke

    def counted(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConversationalRetrievalChain, "invoke", counted)
    app = start_app(monkeypatch, {"current": two_pdf_uploads})
    click(app, "Process")
    assert app.session_state["indexed_documents"], "processing should record the documents"
    assert calls["count"] == 0, "ingestion must not call the LLM"

    ask(app, "What do database indexes use?")
    assert calls["count"] == 1
    assert not app.exception, [e.value for e in app.exception]

    app.run()  # an unrelated rerun must not resubmit the question
    assert calls["count"] == 1


def test_sources_block_is_rendered_with_real_citations(monkeypatch, two_pdf_uploads):
    app = start_app(monkeypatch, {"current": two_pdf_uploads})
    click(app, "Process")
    ask(app, "What do database indexes use?")

    blocks = [block.value for block in app.markdown if "chat-sources" in block.value]
    assert blocks, "an answer should render a sources block"
    rendered = blocks[-1]
    expected = app.session_state["answer_sources"][-1]

    assert {"source": "A.pdf", "page": 2} in expected
    assert "A.pdf" in rendered and "Page 2" in rendered
    assert rendered.count("<li>") == len(expected), "citations must come from retrieved chunks only"


def test_reprocessing_clears_transcript_and_replaces_documents(monkeypatch):
    uploads = {"current": [FakeUpload(make_pdf([A_PAGE_1, A_PAGE_2]), "A.pdf")]}
    app = start_app(monkeypatch, uploads)
    click(app, "Process")
    ask(app, "What do database indexes use?")
    assert app.session_state["chat_history"], "expected a transcript after asking"
    assert app.session_state["answer_sources"]

    uploads["current"] = [FakeUpload(make_pdf([B_PAGE_1, B_PAGE_2]), "B.pdf")]
    click(app, "Process")

    assert app.session_state["chat_history"] == []
    assert app.session_state["answer_sources"] == []
    assert app.session_state["last_retrieved"] == []
    assert [entry["source"] for entry in app.session_state["indexed_documents"]] == ["B.pdf"]

    documents = app.session_state["conversation"].retriever.invoke("database transactions")
    assert documents
    assert all(document.metadata["source"] == "B.pdf" for document in documents)


def test_failed_ingestion_clears_stale_state(monkeypatch):
    uploads = {"current": [FakeUpload(make_pdf([A_PAGE_1]), "A.pdf")]}
    app = start_app(monkeypatch, uploads)
    click(app, "Process")
    ask(app, "What guarantees do database transactions have?")
    assert app.session_state["chat_history"]

    uploads["current"] = [FakeUpload(b"not a pdf at all", "broken.pdf")]
    click(app, "Process")

    assert app.session_state["conversation"] is None
    assert app.session_state["vectorstore"] is None
    assert app.session_state["chat_history"] == []
    assert app.session_state["answer_sources"] == []
    assert app.session_state["indexed_documents"] == []
    assert any("No readable text" in error.value for error in app.error)


def test_script_payloads_are_escaped_in_the_transcript(monkeypatch):
    app = start_app(monkeypatch, {"current": []})
    app.session_state["conversation"] = ScriptedConversation()
    ask(app, "hello")
    assert all("<script>" not in block.value for block in app.markdown)
    assert all("<img src=x" not in block.value for block in app.markdown)
    assert any("&lt;script&gt;" in block.value for block in app.markdown)


def test_asking_without_a_document_warns_and_does_not_call_the_llm(monkeypatch):
    calls = {"count": 0}
    original = ConversationalRetrievalChain.invoke

    def counted(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConversationalRetrievalChain, "invoke", counted)
    app = start_app(monkeypatch, {"current": []})
    ask(app, "anything?")
    assert calls["count"] == 0
    assert any("upload and process a PDF" in warning.value for warning in app.warning)


def test_stylesheet_stays_valid_and_styles_the_sources():
    css = chatpdf.css
    assert "align:" not in css, "the invalid align declaration came back"
    assert "justify-content: flex-end" in css
    assert "white-space: pre-wrap" in css
    assert "background-color: #2b313e" in css and "background-color: #475063" in css
    assert ".chat-sources" in css


# --------------------------------------------------------------------------- #
# Live tests against the real Ollama server and llama3.2 (skipped when absent)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live_chain(two_pdf_uploads):
    page_documents, _ = chatpdf.extract_pdf_pages(two_pdf_uploads)
    store = chatpdf.get_vectorstore(chatpdf.build_chunks(page_documents))
    return chatpdf.get_conversationchain(store)


@live
def test_live_answer_comes_from_the_right_document_with_a_citation(live_chain):
    response = live_chain.invoke({"question": "What do database indexes use?"})
    sources = chatpdf.collect_sources(response["source_documents"])
    assert {"source": "A.pdf", "page": 2} in sources, "the DBMS page must be retrieved"
    answer = response["answer"].lower()
    assert "tree" in answer or "index" in answer


@live
def test_live_unrelated_question_is_not_fabricated(live_chain):
    response = live_chain.invoke({"question": "What is the capital city of France?"})
    answer = response["answer"].lower()
    not_found_phrases = (
        "could not find",
        "not found",
        "no information",
        "not mentioned",
        "not in the uploaded",
        "does not contain",
        "no relevant",
    )
    assert "paris" not in answer, "the model answered from outside the documents"
    assert any(phrase in answer for phrase in not_found_phrases), response["answer"]


@live
def test_live_follow_up_keeps_conversational_retrieval(live_chain):
    first = live_chain.invoke({"question": "What do database indexes use?"})
    assert {"source": "A.pdf", "page": 2} in chatpdf.collect_sources(first["source_documents"])

    follow_up = live_chain.invoke({"question": "and why does that help range queries?"})
    sources = chatpdf.collect_sources(follow_up["source_documents"])
    assert {"source": "A.pdf", "page": 2} in sources, "the follow-up lost the original context"
    assert follow_up["answer"].strip()

