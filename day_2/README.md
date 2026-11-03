# Day 2 — LangGraph, Reliable Agents

On Day 1 the assistant was a straight line: every question went
`retrieve → answer`. That is fine until questions are **different** from each other.

- A math question does not need the handbook — it needs a calculator.
- A complaint about grades should **not** be answered automatically — a human should check.

A straight chain cannot make those choices. A **graph** can.

Run the steps in order:

```bash
python day_2/01_langgraph_basics.py       # the shape: state, nodes, edges
python day_2/02_knowledge_assistant.py    # the assistant as a graph (type 'quit')
streamlit run day_2/app.py                # a web chat page to demo

# OPTIONAL stretch — a real local vector database (needs extra packages):
uv sync --extra vectordb                  # one-time install
python day_2/03_vector_db_optional.py     # BM25 -> embeddings, saved to disk
```

---

## Step 1 — The shape (`01_langgraph_basics.py`)
Three ideas:

- **State** — a shared dictionary passed from step to step.
- **Nodes** — functions that read the state and return updates.
- **Edges** — the arrows connecting the nodes.

```mermaid
flowchart LR
    START((start)) --> measure[measure] --> report[report] --> END((end))
```

## Step 2 — The assistant as a graph (`02_knowledge_assistant.py`)
The **same** assistant as Day 1, but a **router** now sends each question down the
right path, and sensitive questions **pause for a human**:

```mermaid
flowchart TD
    START((start)) --> router{router<br/>classify intent & risk}
    router -->|knowledge| retrieve[retrieve from docs] --> answer[grounded answer]
    router -->|math| calc[calculate] --> E((end))
    router -->|sensitive| human[human review<br/>approve / correct] --> E
    answer --> E
```

Try these three questions to see each path:

| Question | Path it takes |
|---|---|
| `What are the library hours?` | knowledge → retrieve |
| `What is 15 * 4?` | math → calculate |
| `I think my grade is wrong, what do I do?` | sensitive → **human review** |

## Step 3 — A web page (`app.py`)
A simple **Streamlit** chat page so teams can demo the assistant without building a
full frontend. It shows the answer **and the sources used**.

The sidebar has a **Retrieval method** switch — *Keyword (BM25)* or *Vector DB (Chroma)* —
so you can flip between the two live and see that the rest of the app is unchanged.

## Step 3b — OPTIONAL: a real vector database (`03_vector_db_optional.py`)
Everything so far used **BM25**, which matches *words*. A vector database matches
*meaning*: each chunk becomes a list of numbers (an **embedding**), and search finds
the closest ones.

```bash
uv sync --extra vectordb                  # installs langchain-chroma + fastembed
python day_2/03_vector_db_optional.py
```

- A small embedding model downloads **once** (~130MB, no API key).
- The database is **saved to disk** in `day_2/chroma_db/` — the 2nd run loads instantly.
- **Only the retriever line changes** vs `day_1/03_rag_hint.py`:
  `BM25Retriever.from_documents(...)` → `db.as_retriever(...)`. That is the whole point —
  swap one node, keep the graph.

```mermaid
flowchart LR
    D[data/*.md] --> C[split into chunks]
    C --> E[embed each chunk]
    E --> V[(Chroma vector DB<br/>day_2/chroma_db/)]
    Q[question] --> QE[embed question]
    QE --> V
    V --> R[closest chunks<br/>+ sources]
```

---

## Chain vs Graph — side by side

```mermaid
flowchart TB
    subgraph Day1[Day 1 · Chain · same path every time]
        direction LR
        a[question] --> b[retrieve] --> c[answer]
    end
    subgraph Day2[Day 2 · Graph · chooses a path]
        direction LR
        d[question] --> e{router}
        e -->|knowledge| f[retrieve] --> g[answer]
        e -->|math| h[calculate] --> g
        e -->|sensitive| i[human review] --> g
    end
```

**That is the lesson of the seminar:** start simple with a chain, then reach for a
graph when you need decisions, tools, retries, and human review.
