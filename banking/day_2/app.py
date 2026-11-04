"""
Banking Seminar · Day 2 — Streamlit app: the bank assistant as a web chat.

Same ideas as 02_banking_assistant.py, but on a web page:
  * check balance / deposit        -> done right away
  * withdraw                       -> HUMAN-IN-THE-LOOP: Approve / Decline buttons
  * policy question                -> answered from the bank's docs (RAG)
  * sensitive (fraud, lost card)   -> forwarded to a human

Run it:   streamlit run banking/day_2/app.py
"""

import json
import os
from pathlib import Path
from typing import Literal

import warnings

warnings.filterwarnings("ignore", message=r".*langchain-community.*sunset.*")

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[1] / "data"
load_dotenv(ROOT / ".env")

# The account details live in a plain JSON file (no database). The terminal
# scripts and this web app share the SAME file, so the balance persists.
ACCOUNT_FILE = DATA / "account.json"


def load_account() -> dict:
    """Read the account details from the JSON file."""
    return json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))


def save_account(account: dict) -> None:
    """Write the account details back to the JSON file (saves to disk)."""
    ACCOUNT_FILE.write_text(json.dumps(account, indent=2), encoding="utf-8")


class Intent(BaseModel):
    action: Literal["balance", "deposit", "withdraw", "other"] = Field(
        description="The kind of banking action the user wants"
    )
    amount: float = Field(description="The amount of money, or 0 if none")


# Build the model + retriever ONCE, not on every message.
@st.cache_resource
def setup():
    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )
    docs = []
    for md in DATA.glob("*.md"):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 2
    return model, retriever


model, retriever = setup()

st.title("🏦 Sandbox Bank Assistant")
st.caption("Check your balance, deposit, withdraw (with approval), or ask about our policies.")

# --- Session state: survives each rerun ---
if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None  # a withdrawal amount waiting for approval

# The balance is NOT kept in session state -- it is read fresh from the JSON
# file, so it is always correct and shared with the terminal scripts.
account = load_account()
st.sidebar.metric("Current balance", f"UGX {account['balance']:,.0f}")
st.sidebar.caption(f"{account['name']} · {account['account_number']}")


def route_of(message: str) -> str:
    decision = model.invoke(
        [
            (
                "system",
                "Classify the message into ONE word: 'transaction' (balance, "
                "deposit or withdraw), 'sensitive' (fraud, lost card, dispute, "
                "complaint), or 'policy' (rules, fees, limits, interest, hours). "
                "Reply with ONLY that word.",
            ),
            ("user", message),
        ]
    ).content.lower()
    return next((r for r in ("sensitive", "transaction", "policy") if r in decision), "policy")


def policy_answer(message: str) -> str:
    found = retriever.invoke(message)
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}" for d in found
    )
    return model.invoke(
        [
            (
                "system",
                "Answer using ONLY the context and cite the source file. If the "
                "answer is not there, say you could not find it in the bank's "
                "policies.\n\n" + context,
            ),
            ("user", message),
        ]
    ).content


# Show the conversation so far.
for role, text in st.session_state.history:
    st.chat_message(role).write(text)

# If a withdrawal is waiting, show the HUMAN approval buttons.
if st.session_state.pending is not None:
    amount = st.session_state.pending
    st.warning(f"Approval needed: withdraw UGX {amount:,.0f}?")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve", key="approve"):
        acct = load_account()
        acct["balance"] -= amount
        save_account(acct)
        st.session_state.pending = None
        st.session_state.history.append(
            ("assistant", f"Withdrew UGX {amount:,.0f}. New balance: UGX {acct['balance']:,.0f}.")
        )
        st.rerun()
    if col2.button("❌ Decline", key="decline"):
        st.session_state.pending = None
        st.session_state.history.append(
            ("assistant", "The withdrawal was declined by the human reviewer.")
        )
        st.rerun()

# Only accept new messages when no approval is pending.
question = st.chat_input("Ask the bank assistant...", disabled=st.session_state.pending is not None)
if question:
    st.session_state.history.append(("user", question))
    route = route_of(question)

    if route == "transaction":
        intent = model.with_structured_output(Intent).invoke(
            "Extract the banking action and amount from this message: " + question
        )
        action, amount = intent.action.lower(), intent.amount
        acct = load_account()
        if "balance" in action:
            reply = f"Your balance is UGX {acct['balance']:,.0f}."
        elif "deposit" in action:
            if amount <= 0:
                reply = "Please tell me a positive amount to deposit."
            else:
                acct["balance"] += amount
                save_account(acct)
                reply = f"Deposited UGX {amount:,.0f}. New balance: UGX {acct['balance']:,.0f}."
        elif "withdraw" in action:
            if amount <= 0:
                reply = "Please tell me a positive amount to withdraw."
            elif amount > acct["balance"]:
                reply = f"Insufficient funds. Your balance is only UGX {acct['balance']:,.0f}."
            else:
                st.session_state.pending = amount
                reply = f"A withdrawal of UGX {amount:,.0f} needs your approval below."
        else:
            reply = "I can check your balance, make a deposit, or make a withdrawal."
    elif route == "sensitive":
        reply = "This looks sensitive, so I've forwarded it to our support team. A human will follow up."
    else:
        reply = policy_answer(question)

    st.session_state.history.append(("assistant", reply))
    st.rerun()
