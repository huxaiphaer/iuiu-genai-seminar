"""
Banking Seminar · Day 2 (OPTIONAL stretch) — A real local vector database.

So far policy search used BM25, which matches WORDS. A vector database matches
MEANING: each chunk of text becomes a list of numbers (an "embedding"), and
search finds the chunks whose numbers are closest to the question's.

The big teaching point:
  Compare this to banking/day_1/03_rag_policies.py. Only the RETRIEVER changed.
  Everything after `retriever.invoke(question)` is exactly the same shape --
  "swap one piece, keep the rest".

One-time install (kept separate so the core workshop stays zero-setup):
    uv sync --extra vectordb

Run it:   python banking/day_2/03_vector_db_optional.py
"""

import os
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message=r".*langchain-community.*sunset.*")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[1] / "data"
load_dotenv(ROOT / ".env")

# Where the local vector database lives on disk. Delete this folder to rebuild.
DB_DIR = Path(__file__).resolve().parents[1] / "chroma_db"

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# The embedding model runs locally and downloads a small model the FIRST time
# only -- no API key needed.
embeddings = FastEmbedEmbeddings()

# --- Build the database once, then reuse it from disk ---
if DB_DIR.exists():
    print("Loading the existing vector database from disk (fast)...")
    db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
else:
    print("Building the vector database for the first time (downloads a model once)...")
    docs = []
    for md_file in DATA.glob("*.md"):
        docs.extend(TextLoader(str(md_file), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=str(DB_DIR))
    print(f"Saved the database to: {DB_DIR}")

# This one line is the ONLY real difference from the BM25 version.
retriever = db.as_retriever(search_kwargs={"k": 2})


def answer(question: str) -> str:
    # Everything below is IDENTICAL to the BM25 version -- that is the point.
    found = retriever.invoke(question)
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}" for d in found
    )
    messages = [
        (
            "system",
            "Answer using ONLY the context below and cite the source file. If "
            "the answer is not there, say: 'I could not find this in the bank's "
            "policies.'\n\n" + context,
        ),
        ("user", question),
    ]
    return model.invoke(messages).content


if __name__ == "__main__":
    # A question worded DIFFERENTLY from the documents. Vector search can still
    # find it by meaning, where exact-keyword BM25 might miss.
    EXAMPLE = "How much can I take out in a single day?"
    q = input(f"Ask about the bank's policies [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE
    print("\nQ:", q)
    print("A:", answer(q))
