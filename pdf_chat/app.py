import html
from io import BytesIO
from typing import Any, Optional

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

from htmlTemplates import css, bot_template, sources_template, user_template


# --- configuration (values unchanged from the working implementation) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
RETRIEVER_K = 4
# FAISS fetches this many candidates before applying a metadata filter
FILTER_FETCH_K = 50
DEBUG_PREVIEW_CHARS = 300
NOT_FOUND_MESSAGE = "I could not find that information in the uploaded documents."


CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)

# Every retrieved chunk is labelled with its document and page, so the model can
# see where each piece of context came from.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {source}, Page: {page}]\n{page_content}"
)

# Built by concatenation so that {context} stays a literal prompt placeholder.
ANSWER_SYSTEM_PROMPT = (
    "You answer questions about the user's uploaded PDF documents.\n"
    "Use ONLY the retrieved context below. Every context block starts with "
    "[Source: <file>, Page: <number>].\n"
    'If the retrieved context does not contain the answer, reply exactly: "'
    + NOT_FOUND_MESSAGE
    + '"\n'
    "Never invent facts, file names or page numbers. Keep answers to three "
    "sentences or fewer unless the question asks for a list or more detail.\n\n"
    "Retrieved context:\n{context}"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_SYSTEM_PROMPT), ("human", "{question}")]
)


def build_text_splitter():
    """The existing chunk configuration (chunk_size=1000, chunk_overlap=200)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def source_labels(uploaded_files):
    """Give every upload its own (document_id, display label).

    Two uploads can carry the same file name; the second one is labelled
    "name (2)" so citations and the document filter stay unambiguous.
    """
    seen, labels = {}, []
    for index, upload in enumerate(uploaded_files, start=1):
        name = getattr(upload, "name", None) or f"document-{index}.pdf"
        seen[name] = seen.get(name, 0) + 1
        document_id = f"doc{index}"
        labels.append((document_id, name if seen[name] == 1 else f"{name} ({seen[name]})"))
    return labels


def extract_pdf_pages(uploaded_files):
    """Extract text page by page.

    Returns ``(page_documents, report)``: one Document per successfully extracted
    page, plus a per-file report so the UI can state exactly what was indexed.
    A page that raises is recorded and skipped - it never aborts the file - and
    pages without text are never reported as indexed.
    """
    pages, report = [], []
    for (document_id, label), upload in zip(source_labels(uploaded_files), uploaded_files):
        entry = {
            "document_id": document_id,
            "source": label,
            "pages_indexed": 0,
            "empty_pages": [],
            "failed_pages": [],
            "error": None,
        }
        report.append(entry)

        try:
            if hasattr(upload, "seek"):
                upload.seek(0)  # re-processing the same upload must still work
            data = upload.read() if hasattr(upload, "read") else upload
            reader = PdfReader(BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
        except Exception as e:
            entry["error"] = str(e)
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                entry["failed_pages"].append(page_number)
                continue
            if not page_text.strip():
                entry["empty_pages"].append(page_number)
                continue
            entry["pages_indexed"] += 1
            pages.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": label,
                        "page": page_number,
                        "document_id": document_id,
                    },
                )
            )
    return pages, report


def build_chunks(page_documents):
    """Split every page into chunks, keeping source/page metadata on each chunk."""
    splitter = build_text_splitter()
    counts: dict = {}
    chunks = []
    for page in page_documents:
        document_id = page.metadata["document_id"]
        for piece in splitter.split_text(page.page_content):
            text = piece.strip()
            if not text:
                continue  # never index empty chunks
            counts[document_id] = counts.get(document_id, 0) + 1
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **page.metadata,
                        "chunk_id": f"{document_id}-c{counts[document_id]}",
                    },
                )
            )
    return chunks


def get_vectorstore(chunks):
    if not chunks:
        raise ValueError("No text chunks created from the uploaded PDF(s).")

    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "HuggingFaceEmbeddings not available. Install with: "
            "pip install langchain-huggingface sentence-transformers"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


class ScopedRetriever(BaseRetriever):
    """FAISS retriever that can be limited to a single uploaded document.

    Retrieval is the same FAISS similarity search as before (same k, same ranking);
    the raw L2 distance is kept on each document so the debug panel can show it.
    """

    vectorstore: Any
    k: int = RETRIEVER_K
    document_id: Optional[str] = None

    def _get_relevant_documents(self, query, *, run_manager=None):
        filter_ = {"document_id": self.document_id} if self.document_id else None
        # FAISS applies the filter after fetching candidates, so fetch more when scoped.
        fetch_k = FILTER_FETCH_K if filter_ else self.k
        documents = []
        for document, score in self.vectorstore.similarity_search_with_score(
            query, k=self.k, filter=filter_, fetch_k=fetch_k
        ):
            document.metadata = {**document.metadata, "score": float(score)}
            documents.append(document)
        return documents


def get_conversationchain(vectorstore, document_id=None, history=None):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    if history:
        # Changing the search scope rebuilds the chain, so carry the conversation over.
        memory.chat_memory.add_messages([m for m in history if isinstance(m, BaseMessage)])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ScopedRetriever(
            vectorstore=vectorstore, k=RETRIEVER_K, document_id=document_id
        ),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        combine_docs_chain_kwargs={
            "prompt": ANSWER_PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
            "document_separator": "\n\n",
        },
        return_source_documents=True,
        memory=memory,
    )


def message_role(message_type):
    """Map a LangChain message type to the role used by the templates."""
    if message_type in {"human", "user"}:
        return "user"
    if message_type in {"ai", "assistant", "bot"}:
        return "assistant"
    # system/tool/unknown messages are shown as assistant output
    return "assistant"


def expand_history(history):
    """Flatten the stored transcript into ``(role, text)`` pairs.

    The role always comes from the message itself (``HumanMessage`` / ``AIMessage``
    ``.type``) and never from the position of the message in the list. The
    ``(human, ai)`` tuple shape that ConversationBufferMemory produces when
    ``return_messages=False`` is still supported as a fallback.
    """
    pairs = []
    for entry in history or []:
        if (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and all(isinstance(part, str) for part in entry)
        ):
            pairs.append(("user", entry[0]))
            pairs.append(("assistant", entry[1]))
        elif isinstance(entry, (tuple, list)):
            pairs.extend(expand_history(entry))
        else:
            text = getattr(entry, "content", entry)
            kind = getattr(entry, "type", None)
            if kind is None and isinstance(entry, BaseMessage):
                kind = entry.type
            pairs.append((message_role(kind), str(text)))
    return pairs


def collect_sources(source_documents):
    """Unique ``(source, page)`` pairs taken from the retrieved documents.

    Only documents the retriever actually returned are reported - nothing is
    invented - and repeated document/page pairs are collapsed, keeping the order
    of retrieval.
    """
    seen, sources = set(), []
    for document in source_documents or []:
        metadata = getattr(document, "metadata", None) or {}
        source = metadata.get("source") or "unknown document"
        page = metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": page})
    return sources


def format_sources(sources):
    """Render the sources block for one answer (all values escaped)."""
    if not sources:
        return ""
    items = "".join(
        "<li>📄 <code>{}</code> — Page {}</li>".format(
            html.escape(str(source["source"])),
            html.escape("unknown" if source["page"] is None else str(source["page"])),
        )
        for source in sources
    )
    return sources_template.replace("{{ITEMS}}", items)


def render_chat_history():
    """Draw the whole stored transcript.

    Runs on every Streamlit rerun and never submits a question, so the transcript
    stays visible without triggering another LLM call.
    """
    assistant_index = 0
    for role, text in expand_history(st.session_state.chat_history):
        template = user_template if role == "user" else bot_template
        # Only the static template markup is trusted: PDF/LLM text is escaped so
        # it can never inject HTML into the page.
        st.write(template.replace("{{MSG}}", html.escape(text)), unsafe_allow_html=True)

        if role == "user":
            continue
        stored = st.session_state.answer_sources
        if assistant_index < len(stored):
            block = format_sources(stored[assistant_index])
            if block:
                st.write(block, unsafe_allow_html=True)
        assistant_index += 1


def render_debug_panel():
    """Collapsed developer view of the chunks retrieved for the last answer."""
    chunks = st.session_state.last_retrieved
    with st.expander("🔍 Retrieval debug (developer)", expanded=False):
        if not chunks:
            st.caption("Ask a question to see what the retriever returned.")
            return
        st.caption(f"Retrieved chunks for the last answer: {len(chunks)}")
        st.table(
            {
                "#": [chunk["rank"] for chunk in chunks],
                "Document": [chunk["source"] for chunk in chunks],
                "Page": [chunk["page"] for chunk in chunks],
                "Chunk": [chunk["chunk_id"] for chunk in chunks],
                "Distance (lower = closer)": [chunk["distance"] for chunk in chunks],
            }
        )
        if st.checkbox(
            f"Show first {DEBUG_PREVIEW_CHARS} characters of each chunk",
            key="debug_show_preview",
        ):
            for chunk in chunks:
                st.text(chunk["preview"])


def ask_question(question):
    """Send exactly one question to the chain and store the updated transcript."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
        return

    try:
        with st.spinner("Searching your documents..."):
            response = st.session_state.conversation.invoke({"question": question})
    except Exception as e:
        st.error(f"Error while answering: {e}")
        return

    source_documents = response.get("source_documents") or []
    history = response.get("chat_history")
    if not history:
        # Defensive fallback: keep the transcript in sync even if the chain does
        # not return a chat history.
        history = list(st.session_state.chat_history) + [
            HumanMessage(content=question),
            AIMessage(content=response.get("answer", "")),
        ]

    st.session_state.chat_history = history
    # One entry per answer, so the transcript can be re-rendered on every rerun.
    st.session_state.answer_sources.append(collect_sources(source_documents))
    st.session_state.last_retrieved = [
        {
            "rank": rank,
            "source": (getattr(document, "metadata", None) or {}).get("source", "unknown"),
            "page": (getattr(document, "metadata", None) or {}).get("page"),
            "chunk_id": (getattr(document, "metadata", None) or {}).get("chunk_id", "-"),
            "distance": round(
                float((getattr(document, "metadata", None) or {}).get("score", float("nan"))), 3
            ),
            "preview": getattr(document, "page_content", "")[:DEBUG_PREVIEW_CHARS],
        }
        for rank, document in enumerate(source_documents, start=1)
    ]


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("answer_sources", [])
    st.session_state.setdefault("last_retrieved", [])
    st.session_state.setdefault("indexed_documents", [])
    st.session_state.setdefault("search_document_id", "all")
    st.session_state.setdefault("index_version", 0)

    st.header("Chat with multiple PDFs :books:")
    st.caption("Running locally with Ollama (llama3.2) + HuggingFace embeddings — no API key needed.")

    # A form only submits on an explicit click, so a rerun (any widget
    # interaction) can never resubmit the previous question.
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Ask a question about your document:")
        submitted = st.form_submit_button("Ask")

    if submitted:
        if question.strip():
            ask_question(question.strip())
        else:
            st.warning("Please type a question before submitting.")

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Process"):
            # A new processing attempt always replaces the previous document set,
            # so a failed attempt can never leave the old vector store active.
            st.session_state.conversation = None
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.session_state.answer_sources = []
            st.session_state.last_retrieved = []
            st.session_state.indexed_documents = []
            st.session_state.search_document_id = "all"

            if not docs:
                st.warning("Please upload at least one PDF.")
                return

            try:
                with st.spinner("Processing... (first run downloads the embedding model ~90MB)"):
                    page_documents, report = extract_pdf_pages(docs)
                    if not page_documents:
                        st.error(
                            "No readable text found in the uploaded PDF(s). "
                            "It may be a scanned image PDF."
                        )
                        return

                    text_chunks = build_chunks(page_documents)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = get_conversationchain(vectorstore)
            except Exception as e:
                st.error(f"Could not process the uploaded PDF(s): {e}")
                return

            chunks_per_document: dict = {}
            for chunk in text_chunks:
                key = chunk.metadata["document_id"]
                chunks_per_document[key] = chunks_per_document.get(key, 0) + 1

            # A new index generation gets a fresh scope widget, so re-processing
            # can never leave a stale document selection behind.
            st.session_state.index_version += 1
            st.session_state.indexed_documents = [
                {
                    "document_id": entry["document_id"],
                    "source": entry["source"],
                    "pages_indexed": entry["pages_indexed"],
                    "chunks": chunks_per_document.get(entry["document_id"], 0),
                    "failed_pages": entry["failed_pages"],
                    "empty_pages": entry["empty_pages"],
                    "error": entry["error"],
                }
                for entry in report
            ]
            st.success(
                f"Processed {len(page_documents)} page(s) into {len(text_chunks)} chunks. "
                "Ask away!"
            )

        indexed = st.session_state.indexed_documents
        if indexed:
            st.markdown(f"**Indexed documents ({len(indexed)})**")
            for entry in indexed:
                st.markdown(f"- ✓ `{entry['source']}`")
                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;{entry['pages_indexed']} page(s), {entry['chunks']} chunk(s)")
                if entry["failed_pages"]:
                    st.warning(
                        f"⚠️ `{entry['source']}`: {len(entry['failed_pages'])} page(s) could not "
                        f"be extracted (page {', '.join(map(str, entry['failed_pages']))})."
                    )
                if entry["empty_pages"]:
                    st.caption(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;ℹ️ {len(entry['empty_pages'])} page(s) had no "
                        f"extractable text and were not indexed "
                        f"(page {', '.join(map(str, entry['empty_pages']))})."
                    )
                if entry["error"]:
                    st.error(f"⚠️ `{entry['source']}` was not indexed: {entry['error']}")

            st.caption(
                f"{len(indexed)} document(s) and "
                f"{sum(entry['chunks'] for entry in indexed)} chunk(s) indexed."
            )

            scope_options = [("all", "All documents")] + [
                (entry["document_id"], entry["source"]) for entry in indexed
            ]
            scope_index = next(
                (
                    position
                    for position, option in enumerate(scope_options)
                    if option[0] == st.session_state.search_document_id
                ),
                0,
            )
            scope = st.selectbox(
                "Search in:",
                scope_options,
                index=scope_index,
                format_func=lambda option: option[1],
                key=f"search_scope_{st.session_state.index_version}",
            )
            if scope[0] != st.session_state.search_document_id:
                st.session_state.search_document_id = scope[0]
                if st.session_state.vectorstore is not None:
                    # Rebuild the retriever for the new scope, keeping the conversation.
                    st.session_state.conversation = get_conversationchain(
                        st.session_state.vectorstore,
                        document_id=None if scope[0] == "all" else scope[0],
                        history=st.session_state.chat_history,
                    )

    # The transcript is rendered on every rerun, always from the current
    # session state and completely outside of the question handling. Rendering
    # last means a "Process" click that clears/rebuilds the document state is
    # reflected in the same run (no stale transcript from the previous PDFs).
    render_chat_history()
    render_debug_panel()


if __name__ == "__main__":
    main()