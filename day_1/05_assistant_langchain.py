"""
Day 1 (LangChain) - Step 5: The University Knowledge Assistant, LangChain style.

This is the Day 1 capstone. It is a LINEAR pipeline -- a straight line:

        question  ->  retrieve  ->  grounded answer

EVERY question follows the exact same path. There is no routing, no tool
choice, no human review. That is the strength AND the limit of a plain chain.

On Day 2 we rebuild THIS SAME assistant with LangGraph so you can see the
difference:  day_2/02_knowledge_assistant.py

Run it:   python day_1/05_assistant_langchain.py   (type 'quit' to exit)
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

# Load + split + index the approved documents (same three steps as Step 3).
docs = []
for md_file in (ROOT / "data").glob("*.md"):
    docs.extend(TextLoader(str(md_file), encoding="utf-8").load())
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
retriever = BM25Retriever.from_documents(chunks)
retriever.k = 2


def answer(question: str) -> str:
    """The whole pipeline, as one straight line."""
    found = retriever.invoke(question)
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}"
        for d in found
    )
    messages = [
        (
            "system",
            "You are a university assistant. Answer using ONLY the context below "
            "and cite the source file. If the answer is not in the context, say: "
            "'I could not find this in the approved documents.'\n\n" + context,
        ),
        ("user", question),
    ]
    return model.invoke(messages).content


if __name__ == "__main__":
    print("University Knowledge Assistant (LangChain). Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        print("Assistant:", answer(question), "\n")
