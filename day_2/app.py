"""
Day 2 - Streamlit app: a chat page for the University Knowledge Assistant.

This gives teams a simple web page to demo the assistant without building a
full frontend. It uses the Day 1 retrieval idea (keep it readable); wiring in
the full LangGraph router from 02_knowledge_assistant.py is a nice stretch goal.

Run it:   streamlit run day_2/app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


# @st.cache_resource builds the model + retriever ONCE, not on every message.
@st.cache_resource
def setup():
    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )
    docs = []
    for md in (ROOT / "data").glob("*.md"):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 2
    return model, retriever


model, retriever = setup()

st.title("🎓 University Knowledge Assistant")
st.caption("Answers come from approved documents only — with sources.")

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
