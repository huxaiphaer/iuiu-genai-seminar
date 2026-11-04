"""
Day 2 - Streamlit app: a chat page for the University Knowledge Assistant.

This gives teams a simple web page to demo the assistant without building a
full frontend. It uses the Day 1 retrieval idea (keep it readable); wiring in
the full LangGraph router from 02_knowledge_assistant.py is a nice stretch goal.

Run it:   streamlit run day_2/app.py
"""

import os
from pathlib import Path

import warnings

import streamlit as st
from dotenv import load_dotenv

# Keep the terminal clean: hide the "langchain-community is being sunset"
# deprecation notice. The library still works fine.
warnings.filterwarnings("ignore", message=r".*langchain-community.*sunset.*")
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "day_2" / "chroma_db"  # the local vector database folder
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def load_chunks():
    """Read data/*.md and split into small chunks (shared by both methods)."""
    docs = []
    for md in (ROOT / "data").glob("*.md"):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    return RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)


# @st.cache_resource builds each thing ONCE, not on every message.
@st.cache_resource
def get_model():
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )


@st.cache_resource
def get_bm25_retriever():
    """Keyword search (BM25) - zero setup, in memory. This is the default."""
    retriever = BM25Retriever.from_documents(load_chunks())
    retriever.k = 2
    return retriever


@st.cache_resource
def get_vector_retriever():
    """Vector search (Chroma) - matches meaning, saved to disk in chroma_db/.

    Needs the optional extras:  uv sync --extra vectordb
    """
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_chroma import Chroma

    embeddings = FastEmbedEmbeddings()
    if DB_DIR.exists():
        db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    else:
        db = Chroma.from_documents(
            load_chunks(), embedding=embeddings, persist_directory=str(DB_DIR)
        )
    return db.as_retriever(search_kwargs={"k": 2})


model = get_model()

st.title("🎓 University Knowledge Assistant")
st.caption("Answers come from approved documents only — with sources.")

# --- Pick the retrieval method. This is the "swap one node" idea, live. ---
method = st.sidebar.radio(
    "Retrieval method",
    ["Keyword (BM25)", "Vector DB (Chroma)"],
    help="BM25 matches words. Chroma matches meaning (needs: uv sync --extra vectordb).",
)

if method == "Vector DB (Chroma)":
    try:
        retriever = get_vector_retriever()
        st.sidebar.success("Using the Chroma vector database (saved in day_2/chroma_db/).")
    except ImportError:
        st.sidebar.error("Vector DB not installed. Run:  uv sync --extra vectordb")
        st.stop()
else:
    retriever = get_bm25_retriever()
    st.sidebar.info("Using in-memory keyword search (BM25). No database on disk.")

# Keep the chat history in the session so it survives each rerun.
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

question = st.chat_input("Ask about the university...")
if question:
    st.session_state.history.append(("user", question))
    st.chat_message("user").write(question)

    found = retriever.invoke(question)
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}" for d in found
    )
    answer = model.invoke(
        [
            (
                "system",
                "Answer using ONLY the context and cite the source file. If the "
                "answer is not there, say you could not find it.\n\n" + context,
            ),
            ("user", question),
        ]
    ).content

    st.session_state.history.append(("assistant", answer))
    st.chat_message("assistant").write(answer)

    with st.expander("Sources used"):
        for d in found:
            st.markdown(f"**{Path(d.metadata['source']).name}**")
            st.text(d.page_content[:300] + "...")
