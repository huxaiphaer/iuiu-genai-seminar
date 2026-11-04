"""
Banking Seminar · Day 1 (LangChain) — Step 5: Capstone, the STRAIGHT-LINE assistant.

This is a LINEAR pipeline -- a straight line:

        question  ->  retrieve policy  ->  grounded answer

EVERY question follows the exact same path. It can ANSWER questions about the
bank's policies, but it cannot DO anything: no transactions, no routing, no
human review. That is the strength AND the limit of a plain chain.

On Day 2 we rebuild THIS SAME assistant with LangGraph so it can also move
money and pause for a human:  banking/day_2/02_banking_assistant.py

Run it:   python banking/day_1/05_assistant_langchain.py   (type 'quit' to exit)
"""

import os
from pathlib import Path

import warnings

# Keep the workshop terminal clean: hide the "langchain-community is being
# sunset" deprecation notice. The library still works fine.
warnings.filterwarnings("ignore", message=r".*langchain-community.*sunset.*")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[1] / "data"
load_dotenv(ROOT / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# Load + split + index the bank's documents (same three steps as Step 3).
docs = []
for md_file in DATA.glob("*.md"):
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
            "You are a bank assistant. Answer using ONLY the context below and "
            "cite the source file. If the answer is not in the context, say: "
            "'I could not find this in the bank's policies.'\n\n" + context,
        ),
        ("user", question),
    ]
    return model.invoke(messages).content


if __name__ == "__main__":
    print("Sandbox Bank Assistant (LangChain). Type 'quit' to exit.")
    print("Examples:  What is the withdrawal limit?  |  What is the savings interest rate?\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        print("Assistant:", answer(question), "\n")
