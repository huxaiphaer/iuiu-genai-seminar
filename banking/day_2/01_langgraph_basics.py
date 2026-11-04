"""
Banking Seminar · Day 2 (LangGraph) — Step 1: The basics (state, nodes, edges).

A LangChain CHAIN runs in a straight line (Day 1).
A LangGraph GRAPH is built from:
  * STATE  -- a shared dictionary passed from step to step
  * NODES  -- functions that read the state and return updates
  * EDGES  -- arrows that connect the nodes

This tiny two-node graph just shows the shape (no LLM yet): it takes a
withdrawal amount, checks it against the review limit, and reports.

Run it:   python banking/day_2/01_langgraph_basics.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

REVIEW_LIMIT = 500.0  # withdrawals this size or larger get a human review


# 1) STATE: the data that flows through the graph.
class State(TypedDict):
    amount: float
    note: str


# 2) NODES: plain functions. Each returns the fields it wants to change.
def classify(state: State) -> dict:
    note = "large — needs human review" if state["amount"] >= REVIEW_LIMIT else "normal"
    return {"note": note}


def report(state: State) -> dict:
    print(f"A withdrawal of ${state['amount']:.2f} is {state['note']}.")
    return {}


# 3) GRAPH: add the nodes, then connect them with edges.
builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("report", report)
builder.add_edge(START, "classify")     # start -> classify
builder.add_edge("classify", "report")  # classify -> report
builder.add_edge("report", END)         # report -> finish
graph = builder.compile()

if __name__ == "__main__":
    EXAMPLE = "750"
    raw = input(f"Enter a withdrawal amount [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE
    graph.invoke({"amount": float(raw)})
