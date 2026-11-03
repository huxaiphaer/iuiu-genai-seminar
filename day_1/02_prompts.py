"""
Day 1 (LangChain) - Step 2: Prompts, pipelines and structured output.

Two ideas:
  1. A PROMPT TEMPLATE gives the model a fixed role + a blank we fill in.
  2. A CHAIN wires steps together with the pipe  |  operator.
Then a bonus: asking the model for TYPED output instead of free text.

Run it:   python day_1/02_prompts.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# 1) A template: a fixed "system" instruction + a "{topic}" placeholder.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly teacher. Explain in 2-3 simple sentences."),
        ("user", "Explain this to a beginner: {topic}"),
    ]
)

# 2) A chain: prompt -> model -> plain text. Read the | as "then".
chain = prompt | model | StrOutputParser()

print("=== Templated answer ===")
print(chain.invoke({"topic": "why it rains"}))


# 3) Structured output: make the model fill a typed shape, not free text.
#    Here we pull two simple facts out of a sentence into named fields.
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years")


person = model.with_structured_output(Person).invoke(
    "My name is Sara and I am 20 years old."
)

print("\n=== Structured output ===")
print("Name:", person.name)
print("Age:", person.age)
