"""
Closing capstone challenge - A banking assistant with HUMAN-IN-THE-LOOP.

This little project ties the whole seminar together:
  * TOOLS  -- the model can check the balance, deposit, and withdraw
             (the tool idea from Day 1, Step 4).
  * HUMAN  -- before any money LEAVES the account (a withdrawal), the
             assistant PAUSES and a real person must approve it
             (the "human in the loop" idea from Day 2).

Built with a LangChain agent. The balance lives in memory and resets
every run -- there is no real bank, of course.

Run it:   python day_2/04_banking_agent.py   (type 'quit' to exit)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# The account lives in memory and resets every run. Everyone starts with UGX 500,000.
ACCOUNT = {"balance": 500000.0}


# --- Tool 1: read-only, always safe to run -------------------------------
def check_balance() -> str:
    """Return the current account balance."""
    return f"Your balance is UGX {ACCOUNT['balance']:,.0f}."


# --- Tool 2: money coming IN -- allowed automatically ---------------------
def deposit(amount: float) -> str:
    """Add money to the account. 'amount' must be positive."""
    if amount <= 0:
        return "Deposit amount must be positive."
    ACCOUNT["balance"] += amount
    return f"Deposited UGX {amount:,.0f}. New balance: UGX {ACCOUNT['balance']:,.0f}."


# --- Tool 3: money going OUT -- needs a HUMAN to approve ------------------
def withdraw(amount: float) -> str:
    """Take money out of the account. A human must approve it first."""
    if amount <= 0:
        return "Withdrawal amount must be positive."
    if amount > ACCOUNT["balance"]:
        return f"Insufficient funds. Your balance is only UGX {ACCOUNT['balance']:,.0f}."

    # HUMAN IN THE LOOP: pause and let a person approve money leaving.
    print(f"\n[APPROVAL NEEDED] The assistant wants to withdraw UGX {amount:,.0f}.")
    ok = input("Approve this withdrawal? (yes/no): ").strip().lower()
    if ok not in {"yes", "y"}:
        return "The withdrawal was declined by the human reviewer."

    ACCOUNT["balance"] -= amount
    return f"Withdrew UGX {amount:,.0f}. New balance: UGX {ACCOUNT['balance']:,.0f}."


# Hand the model the three tools and tell it how to behave.
agent = create_agent(
    model=model,
    tools=[check_balance, deposit, withdraw],
    system_prompt=(
        "You are a careful banking assistant. Use a tool to check the balance, "
        "deposit, or withdraw money -- never make up numbers. After a tool runs, "
        "tell the user what happened in one short, friendly sentence."
    ),
)


def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("Assistant:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    print("Banking Assistant (LangChain + human-in-the-loop). Type 'quit' to exit.")
    print("Examples:  What's my balance?  |  Deposit 100000  |  Withdraw 200000\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        ask(q)
