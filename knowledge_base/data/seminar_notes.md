# Seminar Notes: Building Generative AI Applications (SAMPLE)

> Sample notes used by the Knowledge Assistant during the seminar.

## What is RAG?

Retrieval-Augmented Generation means the model first RETRIEVES relevant pieces
of approved text, then uses only those pieces to write a grounded answer. This
reduces made-up answers (hallucinations) and lets the assistant cite a source.

## What is a tool?

A tool is a normal function the model is allowed to call, such as a calculator
or a document search. The model decides when to call it, reads the result, and
continues. Tools let a language model take actions, not just produce text.

## What is LangGraph?

LangGraph models an application as a graph of nodes (steps) and edges (arrows).
A shared state object is passed between nodes. Unlike a straight chain, a graph
can make decisions, loop, retry, and pause for human review.

## Human-in-the-loop

For sensitive or low-confidence questions, the assistant should pause and ask a
human to approve or correct the answer before showing it to the student.
