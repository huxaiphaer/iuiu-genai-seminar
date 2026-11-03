"""
Day 2 (OPTIONAL stretch) - Step 3: A real local vector database (Chroma).

So far every script used BM25, which matches WORDS. A vector database matches
MEANING: each chunk of text is turned into a list of numbers (an "embedding"),
and the database finds the chunks whose numbers are closest to the question's.

The big teaching point:
  Compare this file to day_1/03_rag_hint.py. Only the RETRIEVER changed.
  Everything after `retriever.invoke(question)` is exactly the same shape --
  this is the "swap one node, keep the graph" idea from the slides.

What is different from BM25:
  * It uses a small embedding model (downloaded ONCE, ~130MB, no API key).
  * It SAVES to disk in day_2/chroma_db/ -- so the 2nd run loads instantly
    instead of rebuilding. That folder IS the local database.

One-time install (kept separate so the core workshop stays zero-setup):
    uv sync --extra vectordb

Run it:   python day_2/03_vector_db_optional.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Where the local vector database lives on disk. Delete this folder to rebuild.
DB_DIR = ROOT / "day_2" / "chroma_db"

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# The embedding model turns text into numbers. FastEmbed runs locally and
# downloads a small model the FIRST time only -- no API key needed.
embeddings = FastEmbedEmbeddings()

# --- Build the database once, then reuse it from disk ---
if DB_DIR.exists():
    print("Loading the existing vector database from disk (fast)...")
    db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
else:
    print("Building the vector database for the first time (downloads a model once)...")
    docs = []
    for md_file in (ROOT / "data").glob("*.md"):
        docs.extend(TextLoader(str(md_file), encoding="utf-8").load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    db = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=str(DB_DIR)
    )
    print(f"Saved the database to: {DB_DIR}")

# This one line is the ONLY real difference from day_1/03_rag_hint.py.
# BM25Retriever.from_documents(...)  ->  db.as_retriever(...)
retriever = db.as_retriever(search_kwargs={"k": 2})


def answer(question: str) -> str:
    # Everything below is IDENTICAL to the BM25 version -- that is the point.
    found = retriever.invoke(question)

    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}"
        for d in found
    )

    messages = [
        (
            "system",
            "Answer the question using ONLY the context below. "
            "Cite the source file name. If the answer is not in the context, say: "
            "'I could not find this in the approved documents.'\n\n" + context,
        ),
        ("user", question),
    ]
    return model.invoke(messages).content


if __name__ == "__main__":
    # Try a question worded DIFFERENTLY from the documents. Vector search can
    # still find it by meaning, where exact-keyword BM25 might miss.
    EXAMPLE = "When can I use the study space in the evening?"
    q = input(f"Ask about the university [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE
    print("\nQ:", q)
    print("A:", answer(q))
