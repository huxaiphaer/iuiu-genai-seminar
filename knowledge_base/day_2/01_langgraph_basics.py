"""
Day 2 (LangGraph) - Step 1: The basics -- state, nodes, edges.

A LangChain CHAIN runs in a straight line (Day 1).
A LangGraph GRAPH is built from:
  * STATE  -- a shared dictionary passed from step to step
  * NODES  -- functions that read the state and return updates
  * EDGES  -- arrows that connect the nodes

This tiny two-node graph just shows the shape. No LLM yet.

Run it:   python day_2/01_langgraph_basics.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1) STATE: the data that flows through the graph.
class State(TypedDict):
    text: str
    length: int


# 2) NODES: plain functions. Each returns the fields it wants to change.
def measure(state: State) -> dict:
    return {"length": len(state["text"])}


def report(state: State) -> dict:
    print(f"'{state['text']}' has {state['length']} characters.")
    return {}


# 3) GRAPH: add the nodes, then connect them with edges.
builder = StateGraph(State)
builder.add_node("measure", measure)
builder.add_node("report", report)
builder.add_edge(START, "measure")   # start -> measure
builder.add_edge("measure", "report")  # measure -> report
builder.add_edge("report", END)        # report -> finish
graph = builder.compile()

if __name__ == "__main__":
    # Type your own text to measure, or press Enter to use the example hint.
    EXAMPLE = "hello langgraph"
    text = input(f"Text to measure [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE
    graph.invoke({"text": text})
