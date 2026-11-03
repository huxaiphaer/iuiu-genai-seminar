"""
Day 1 (LangChain) - Step 1: Talk to a language model.

The goal of this very first step is tiny: send one message to an LLM and
print its reply. Everything else today builds on this.

Run it:   python day_1/01_chat.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load OPENROUTER_API_KEY and OPENROUTER_MODEL from the .env file at the repo root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# The "chat model" is our connection to the LLM.
# OpenRouter speaks the OpenAI API, so we use ChatOpenAI and simply point it
# at OpenRouter's address. Swapping models later is just changing the name.
model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,  # 0 = focused & repeatable; higher = more creative/random
)

# The simplest possible call: give it text, get a reply back.
# Type your own question, or just press Enter to use the example hint.
EXAMPLE = "Why is the sky blue? Answer in one simple sentence."
question = input(f"Ask the model anything [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE

reply = model.invoke(question)

print("\n" + reply.content)
