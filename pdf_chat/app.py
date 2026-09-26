import html
from io import BytesIO

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

from htmlTemplates import css, bot_template, user_template


CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)


def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        try:
            if hasattr(pdf, "read"):
                pdf_reader = PdfReader(BytesIO(pdf.read()))
            else:
                pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not read {getattr(pdf, 'name', 'a file')}: {e}")
    return text


def get_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(raw_text)


def get_vectorstore(chunks):
    if not chunks:
        raise ValueError("No text chunks created from the uploaded PDF.")

    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "HuggingFaceEmbeddings not available. Install with: "
            "pip install langchain-huggingface sentence-transformers"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


def get_conversationchain(vectorstore):
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
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


def render_chat_history():
    """Draw the whole stored transcript.

    Runs on every Streamlit rerun and never submits a question, so the transcript
    stays visible without triggering another LLM call.
    """
    for role, text in expand_history(st.session_state.chat_history):
        template = user_template if role == "user" else bot_template
        # Only the static template markup is trusted: PDF/LLM text is escaped so
        # it can never inject HTML into the page.
        st.write(template.replace("{{MSG}}", html.escape(text)), unsafe_allow_html=True)


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

    history = response.get("chat_history")
    if not history:
        # Defensive fallback: keep the transcript in sync even if the chain does
        # not return a chat history.
        history = list(st.session_state.chat_history) + [
            HumanMessage(content=question),
            AIMessage(content=response.get("answer", "")),
        ]

    st.session_state.chat_history = history


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("indexed_files", [])
    st.session_state.setdefault("chunk_count", 0)

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

        if st.session_state.indexed_files:
            st.caption(
                "Currently indexed: "
                + ", ".join(st.session_state.indexed_files)
                + f" ({st.session_state.chunk_count} chunks)"
            )

        if st.button("Process"):
            # A new processing attempt always replaces the previous document set,
            # so a failed attempt can never leave the old vector store active.
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.indexed_files = []
            st.session_state.chunk_count = 0

            if not docs:
                st.warning("Please upload at least one PDF.")
                return

            try:
                with st.spinner("Processing... (first run downloads the embedding model ~90MB)"):
                    raw_text = get_pdf_text(docs)
                    if not raw_text.strip():
                        st.error(
                            "No readable text found in the uploaded PDF(s). "
                            "It may be a scanned image PDF."
                        )
                        return

                    text_chunks = get_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversationchain(vectorstore)
            except Exception as e:
                st.error(f"Could not process the uploaded PDF(s): {e}")
                return

            st.session_state.indexed_files = [pdf.name for pdf in docs]
            st.session_state.chunk_count = len(text_chunks)
            st.success(f"Processed {len(text_chunks)} chunks. Ask away!")

    # The transcript is rendered on every rerun, always from the current
    # session state and completely outside of the question handling. Rendering
    # last means a "Process" click that clears/rebuilds the document state is
    # reflected in the same run (no stale transcript from the previous PDFs).
    render_chat_history()


if __name__ == "__main__":
    main()