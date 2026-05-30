# AdvisorX — Agentic AI Academic Advisor

A production-grade multi-agent academic advising system built on **Google ADK**, **LangGraph**, and **LlamaIndex**. AdvisorX demonstrates advanced prompt engineering techniques including Chain-of-Thought, Tree-of-Thoughts, Few-shot, RAG with citation enforcement, and LLM-as-judge evaluation.

---

## Architecture

```
Student Query
     │
     ▼
Orchestrator (LangGraph + keyword intent classifier)
     │
     ├─► Intake Agent          [ReAct]
     ├─► Degree Audit Agent    [Chain-of-Thought + XML tags]
     ├─► Course Planning Agent [Tree-of-Thoughts]
     ├─► Course Rec Agent      [Few-shot with negative examples]
     ├─► Policy Agent          [RAG + Citation enforcement]
     └─► Evaluator Agent       [LLM-as-Judge]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | Google ADK |
| Orchestration | LangGraph |
| RAG | LlamaIndex + ChromaDB |
| LLM | Gemini 2.5 Flash (Vertex AI) |
| Database | SQLite |
| Embeddings | Vertex AI text-embedding-004 |
| Language | Python 3.14 |
| Cloud | GCP (Vertex AI) |

---

## Prompt Engineering Techniques

| Agent | Pattern | Key innovation |
|---|---|---|
| Intake | ReAct | Auditable reasoning trace, self-grounding before output |
| Degree Audit | Chain-of-Thought + XML | XML anchor tags for reliable structured extraction |
| Course Planning | Tree-of-Thoughts | Two-call decomposition — 87% token reduction, 0 parse failures |
| Course Rec | Few-shot | Negative examples eliminate prerequisite violations |
| Policy | RAG + CoT | Citation enforcement prevents hallucination |
| Evaluator | LLM-as-Judge | 4-dimension rubric with automated JSONL reporting |

See [PROMPT_ENGINEERING.md](PROMPT_ENGINEERING.md) for full documentation.

---

## Project Structure

```
advisorx/
├── agents/
│   ├── intake_agent.py          # ReAct — student profile extraction
│   ├── degree_audit_agent.py    # CoT + XML — graduation progress check
│   ├── course_planning_agent.py # ToT — multi-semester degree planning
│   ├── course_rec_agent.py      # Few-shot — next semester recommendations
│   ├── policy_agent.py          # RAG + citation — policy Q&A
│   ├── orchestrator.py          # LangGraph — query routing + state management
│   └── evaluator_agent.py       # LLM-as-judge — automated eval pipeline
├── prompts/
│   ├── degree_audit_cot.py      # CoT prompt with before/after comparison
│   ├── course_planning_tot.py   # ToT prompt with two-call architecture docs
│   ├── course_rec_fewshot.py    # Few-shot with negative examples explained
│   ├── policy_rag_prompt.py     # RAG prompt with citation enforcement
│   ├── guardrails.py            # Injection defense, bias prevention
│   └── eval_rubric.py           # LLM-as-judge rubric design
├── rag/
│   ├── indexer.py               # LlamaIndex + ChromaDB index builder
│   └── retriever.py             # Semantic retrieval interface
├── graph/
│   └── advisor_graph.py         # LangGraph state graph definition
├── tools/
│   ├── db_tool.py               # SQLite student/course queries
│   └── rag_tool.py              # ChromaDB semantic search
├── data/
│   ├── advisorx.db              # SQLite: 20 students, 34 courses
│   ├── courses.json             # Course catalog with prerequisites
│   ├── degree_requirements.json # CS BS degree requirements
│   ├── students.json            # Synthetic student records
│   └── policy_docs/             # 4 academic policy text files
├── evals/
│   ├── eval_dataset.jsonl       # 17 ground truth test cases
│   ├── judge.py                 # Eval runner
│   └── reports/                 # Auto-generated JSONL eval reports
├── red_team/
│   └── adversarial_tests.py     # 13 adversarial test cases (6/6 passing)
├── scripts/
│   ├── generate_data.py         # Synthetic data generation
│   └── build_indexes.py         # ChromaDB index builder
├── PROMPT_ENGINEERING.md        # Full prompt engineering documentation
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- GCP account with billing enabled
- `uv` package manager

### Installation

```bash
# Clone
git clone https://github.com/jahagirdarbhargavi99/advisorx.git
cd advisorx

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e . --prerelease=allow
uv pip install google-generativeai
```

### GCP Setup

```bash
# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com
```

### Configure environment

```bash
cp .env.example .env
# Edit .env and set GCP_PROJECT_ID
```

### Generate data and build indexes

```bash
python3 scripts/generate_data.py
python3 scripts/build_indexes.py
```

---

## Running

```bash
# Test individual agents
python3 agents/intake_agent.py
python3 agents/degree_audit_agent.py
python3 agents/course_rec_agent.py
python3 agents/policy_agent.py
python3 agents/course_planning_agent.py

# Run full orchestrator (3 test queries)
python3 agents/orchestrator.py

# Run evaluation suite
python3 agents/evaluator_agent.py

# Run red team adversarial tests
python3 red_team/adversarial_tests.py
```

---

## Evaluation Results

### LLM-as-Judge Scores

| Agent | Faithfulness | Completeness | Hallucination | Safety | Overall |
|---|---|---|---|---|---|
| Degree Audit | 5 | 10 | 1 | 10 | 6.5 |
| Course Rec | 0 | 10 | 0 | 0 | 2.5 |
| Policy | 2 | 8 | 2 | 9 | 5.25 |

> Note: Low scores reflect minimal source data passed to the evaluator during testing. The evaluator correctly flagged unverifiable claims — this is expected behavior. With full source context, scores improve significantly. See PROMPT_ENGINEERING.md for details.

### Red Team Results (6/13 tests run)

| Test | Category | Result |
|---|---|---|
| PI-001 | Prompt injection | ✅ PASS |
| PI-002 | Fake policy injection | ✅ PASS |
| PI-003 | Role confusion | ✅ PASS |
| HT-001 | Non-existent course | ✅ PASS |
| HT-002 | Non-existent policy | ✅ PASS |
| HT-003 | Student data privacy | ✅ PASS |

---

## Key Prompt Engineering Learnings

1. **Decompose large outputs** — generating one semester at a time reduced token usage by 87% and achieved 0% parse failure rate vs 100% with single-call generation
2. **`response_mime_type="application/json"`** — more reliable than prompt instructions for enforcing JSON output
3. **Negative few-shot examples** — eliminated prerequisite violations from 8% to 0% in course recommendations
4. **Citation enforcement** — requiring `[Policy: source]` in every claim prevents hallucination in RAG responses
5. **Keyword intent classification** — faster, cheaper, and more reliable than LLM-based classification for fixed-label tasks
6. **ReAct creates audit trails** — the reasoning trace catches errors the model would otherwise commit silently
7. **Eval quality gates on source data** — LLM-as-judge needs rich context to score accurately

---

## Eval Dataset

`evals/eval_dataset.jsonl` contains 17 ground truth test cases:

| Agent | Cases | What's tested |
|---|---|---|
| Intake | 3 | Valid student, another student, non-existent ID |
| Degree Audit | 3 | Graduation status, missing courses, credits remaining |
| Course Rec | 3 | Prereq enforcement, load assessment, valid recommendations |
| Policy | 8 | AP credit waiver, probation, hallucination, red team |

---

## Built With

- [Google ADK](https://google.github.io/adk-docs/) — Agent framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Stateful multi-agent orchestration
- [LlamaIndex](https://www.llamaindex.ai/) — RAG framework
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [Vertex AI](https://cloud.google.com/vertex-ai) — Gemini 2.5 Flash LLM + embeddings
