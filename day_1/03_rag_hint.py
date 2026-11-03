"""
Day 1 (LangChain) - Step 3: A first taste of RAG (retrieval).

RAG = Retrieval-Augmented Generation. The shape is always:
  1. Keep a set of approved documents.
  2. RETRIEVE the pieces most relevant to the question.
  3. Put those pieces in the prompt so the model answers from THEM (grounded)
     and can name its source.

We keep it simple here: NO embeddings, NO downloads -- just keyword search
(BM25). It is weaker than embeddings but perfect for seeing the idea.

Run it:   python day_1/03_rag_hint.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# --- Step A: load the approved documents from the data/ folder ---
docs = []
for md_file in (ROOT / "data").glob("*.md"):
    docs.extend(TextLoader(str(md_file), encoding="utf-8").load())

# --- Step B: split long documents into small chunks ---
# Smaller pieces match questions better and make citing a source easier.
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
).split_documents(docs)

# --- Step C: build a keyword retriever over the chunks ---
retriever = BM25Retriever.from_documents(chunks)
retriever.k = 2  # return the 2 best-matching chunks


def answer(question: str) -> str:
    # Retrieve the most relevant chunks for this question.
    found = retriever.invoke(question)

    # Build a context block, tagging each chunk with its source file.
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}"
        for d in found
    )

    # Ground the model: it must use ONLY the context, and admit when unsure.
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
    q = "What are the library opening hours?"
    print("Q:", q)
    print("A:", answer(q))
