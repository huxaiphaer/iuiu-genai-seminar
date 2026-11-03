"""
Day 2 (LangGraph) - Step 2: The SAME assistant, rebuilt as a graph.

Day 1 answered EVERY question with one straight line. A graph can make
DECISIONS. Here the assistant first routes the question, then takes a
different path depending on what it is:

                         +--> [math]      -> calculate --------+
   question -> [router] -+--> [knowledge] -> retrieve -> answer -+-> END
                         +--> [sensitive] -> human review ------+

This is the difference from Day 1: branching, a tool path, and a human
check for risky questions -- all things a straight chain cannot do.

Run it:   python day_2/02_knowledge_assistant.py
"""

import ast
import operator
import os
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# Build the document retriever once (same idea as Day 1).
_docs = []
for _md in (ROOT / "data").glob("*.md"):
    _docs.extend(TextLoader(str(_md), encoding="utf-8").load())
_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(_docs)
retriever = BM25Retriever.from_documents(_chunks)
retriever.k = 2


# A safe mini-calculator (same approach as Day 1, Step 4).
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("bad expression")


# ---------------------------------------------------------------------------
# STATE: the shared data passed between nodes.
# ---------------------------------------------------------------------------
class State(TypedDict):
    question: str
    route: str
    answer: str


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def router(state: State) -> dict:
    """Ask the model to sort the question into one of three buckets."""
    decision = model.invoke(
        [
            (
                "system",
                "Classify the question into exactly ONE word:\n"
                "- 'math' if it is a calculation\n"
                "- 'sensitive' if it is about grades disputes, complaints, money, "
                "personal, legal or anything risky\n"
                "- 'knowledge' for everything else (uses the handbook)\n"
                "Reply with ONLY that one word.",
            ),
            ("user", state["question"]),
        ]
    ).content.lower()

    # Keep only a known label (robust even if the model adds extra words).
    route = next((r for r in ("sensitive", "math", "knowledge") if r in decision), "knowledge")
    print(f"[router] -> {route}")
    return {"route": route}


def retrieve_and_answer(state: State) -> dict:
    """Knowledge path: find relevant text, answer grounded in it, cite sources."""
    found = retriever.invoke(state["question"])
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}" for d in found
    )
    answer = model.invoke(
        [
            (
                "system",
                "Answer using ONLY the context and cite the source file. If the "
                "answer is not there, say: 'I could not find this in the approved "
                "documents.'\n\n" + context,
            ),
            ("user", state["question"]),
        ]
    ).content
    return {"answer": answer}


def calculate(state: State) -> dict:
    """Math path: pull the arithmetic out of the question and compute it."""
    found = re.findall(r"[0-9+\-*/(). ]+", state["question"])
    expr = max(found, key=len).strip() if found else ""
    try:
        return {"answer": f"{expr} = {_eval(ast.parse(expr, mode='eval').body)}"}
    except Exception:
        return {"answer": "I could not work out that calculation."}


def human_review(state: State) -> dict:
    """Sensitive path: pause and let a human approve or correct the answer."""
    print("\n[HUMAN REVIEW NEEDED] This question looks sensitive.")
    print("Question:", state["question"])
    reply = input("Type an approved answer for the student (or Enter to escalate): ").strip()
    final = reply or "This needs a human. I have forwarded it to the faculty office."
    return {"answer": final}


# ---------------------------------------------------------------------------
# EDGES: wire the nodes together, with one DECISION after the router.
# ---------------------------------------------------------------------------
def choose_path(state: State) -> str:
    """Return the name of the next node, based on the route."""
    return {"math": "calculate", "sensitive": "human_review"}.get(state["route"], "retrieve")


builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("retrieve", retrieve_and_answer)
builder.add_node("calculate", calculate)
builder.add_node("human_review", human_review)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", choose_path, ["retrieve", "calculate", "human_review"])
builder.add_edge("retrieve", END)
builder.add_edge("calculate", END)
builder.add_edge("human_review", END)

graph = builder.compile()


if __name__ == "__main__":
    print("University Knowledge Assistant (LangGraph). Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        result = graph.invoke({"question": q})
        print("Assistant:", result["answer"], "\n")
