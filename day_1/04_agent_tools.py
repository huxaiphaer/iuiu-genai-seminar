"""
Day 1 (LangChain) - Step 4: Tools and a simple agent.

A TOOL is just a Python function the model is allowed to call.
An AGENT is a model that can decide WHEN to call a tool, read the result,
and then write the final answer -- looping until it is done.

Run it:   python day_1/04_agent_tools.py
"""

import ast
import operator
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# --- Tool 1: a SAFE calculator ------------------------------------------
# We walk the math expression ourselves instead of using eval(), which is
# dangerous because it can run any code. This only allows + - * / ** and ().
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant):  # a number
        return node.value
    if isinstance(node, ast.BinOp):  # a + b, a * b, ...
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):  # -a
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '3 * (4 + 5)'."""
    return str(_eval(ast.parse(expression, mode="eval").body))


# --- Tool 2: a (pretend) weather lookup ---------------------------------
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's always sunny in {city}!"


# Build the agent: hand it the model, the tools, and how to behave.
agent = create_agent(
    model=model,
    tools=[calculator, get_weather],
    system_prompt="You are a helpful assistant. Use a tool when it helps.",
)


def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("Q:", question)
    print("A:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    ask("What is 12 * (7 + 3)?")           # the model should pick the calculator
    ask("What's the weather in Kampala?")  # ...and here, the weather tool
