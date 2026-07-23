
```markdown
---
title: RAG Document Analyzer
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
app_file: frontend/app.py
pinned: false
---

# Advanced RAG Research Assistant: Transformer Architecture

### System Interface

![alt text](image-1.png)

A production-ready, enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to query complex technical research papers. This project demonstrates an advanced two-stage hybrid retrieval pipeline leveraging Google Gemini 2.5 Flash and a FlashRank Cross-Encoder Reranker, coupled with robust multi-container microservices, enterprise LLMOps logging, asynchronous RAGAS evaluation benchmarks, and automated CI/CD.

Live Demo: https://huggingface.co/spaces/Vivekbiju0/RAG-Document-Analyzer  
Observability Dashboard: https://aws.smith.langchain.com/o/a01e2752-b31e-4cf6-b828-f7a2634b5944/projects/p/defd6317-d455-4f33-847a-b3ff3387a35b  
Production Repo: https://github.com/Vivekbiju0/RAG-Document-Analyzer

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Evaluation & Benchmarking](#evaluation--benchmarking-ragas)
- [Architecture](#architecture)
- [Database Design](#database-design)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
- [License](#license)

---

## Overview

### Motivation
Standard RAG pipelines often suffer from "lost-in-the-middle" context degradation and heavy testing bloat in production environments. This project was engineered to solve these challenges by building a hardened, resource-optimized system that maintains high-precision retrieval on dense, technical datasets under free-tier cloud constraints.

### Objective
To deliver an isolated, dual-stage hybrid retrieval system capable of parsing technical manuscripts, tracking live context-retrieval spans through centralized LLMOps, and offloading heavy evaluation metrics via background tasks without blocking the user experience.

### Learning Outcomes
- Built a multi-process supervisor orchestration layer and backend auto-spawner for single-ingress container environments.
- Implemented a high-precision, two-stage hybrid retrieval framework (Recall vs Precision balancing).
- Configured stateful volume binding and persistence configurations for ChromaDB vector storage.
- Established isolated evaluation frameworks using RAGAS and Google GenAI native clients to decouple production inference from development benchmarking.
- Synchronized parent-child trace execution graphs across API, Retriever, and LLM spans using LangSmith integration.

---

## Features

- **Two-Stage Hybrid Retrieval:** Fetches top candidate nodes ($k=10$) via vector similarity (`models/gemini-embedding-001`) and refines them using a localized Cross-Encoder reranker (`FlashrankRerank`).
- **Automated RAGAS Evaluation Suite:** Evaluates *Faithfulness*, *Answer Relevancy*, *Context Precision*, and *Context Recall* on demand using `gemini-2.5-flash` as an LLM judge, persisting live scores to `metrics.json`.
- **Non-Blocking Background Metrics:** Asynchronous evaluation offloading via FastAPI `BackgroundTasks` queue mapping.
- **Multi-Process Orchestration & Self-Healing:** Runs a FastAPI REST engine and Streamlit frontend concurrently with an auto-spawner process checker.
- **Deep LLMOps Observability:** Centralized span tracking, run tracing, and execution graphs via native `@traceable` hooks mapped to LangSmith.
- **Defensive UI & Payload Architecture:** Handles multi-type LLM payload parsing (lists/dicts/strings) and prevents nested expander UI crashes through popover rendering.
- **Secure Environment Architecture:** Strict routing of keys through ephemeral container memory variables with zero hardcoded credentials.

---

## Tech Stack

### Frontend & Dashboard
- **Streamlit** (Interactive Viewport UX & Control Center)
- **HTML5 / CSS3** (Custom Interface Styling)

### Backend & API
- **FastAPI & Uvicorn** (RESTful API Engine & Async Background Processing)
- **LangChain / LangChain-Community / LangChain-Chroma** (Chain Expression Syntax & Vector Integration)
- **FlashRank** (Lightweight Local Cross-Encoder Reranking Framework)
- **Pydantic** (API Payload Validation)

### Vector Database & AI Core
- **ChromaDB** (Persistent Local Vector Storage Engine)
- **Google Generative AI SDK** (`google-genai` & `langchain-google-genai`)
- **LLM Engine:** Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **Embedding Engine:** Google Gemini Embeddings (`models/gemini-embedding-001`)

### Evaluation & Observability
- **RAGAS Framework** (Automated RAG Metric Benchmarking)
- **Pandas & Hugging Face Datasets** (Metric Aggregation & Reporting)
- **LangSmith** (Centralized Trace & Latency Logging)

### Infrastructure & DevOps
- **Docker & Docker Compose** (Containerization Stack)
- **GitHub Actions** (Automated CI/CD Pipeline)

---

## 🧪 Evaluation & Benchmarking (RAGAS)

The application features an automated RAGAS benchmark module (`src/evaluation_utils.py`) that evaluates generated system responses against ground-truth technical test sets.

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Faithfulness** | **100%** | Measures factual consistency; verifies answers are strictly grounded in retrieved context (zero hallucination). |
| **Answer Relevancy** | **80.1%** | Measures how directly and completely the generated output addresses the input prompt. |
| **Context Precision & Recall** | *Tracked* | Evaluates the signal-to-noise ratio and completeness of retrieved vector chunks. |

> **Note:** Benchmarks can be triggered on demand via the Streamlit Control Center. They run asynchronously via FastAPI `BackgroundTasks`, saving structured summary outputs to `metrics.json` and detailed CSV reports to `data/processed/evaluation_report.csv`.

---

## Architecture

### 1. Production Workflow


```

```
   ┌─────────────────────────────────────────────────────────┐
   │                 Hugging Face Spaces (Pod)               │
   │                                                         │
   │     ┌─────────────────────────────────────────────┐     │
   │     │        Supervisor Process Manager           │     │
   │     └──────┬──────────────────────────────┬───────┘     │
   │            │                              │             │
   │            ▼                              ▼             │
   │ ┌──────────────────────┐      ┌───────────────────────┐ │
   │ │   Streamlit Frontend │      │    FastAPI Backend    │ │
   │ │     (Port 7860)      │      │      (Port 8000)      │ │
   │ └──────────┬───────────┘      └───────────┬───────────┘ │
   └────────────┼──────────────────────────────┼─────────────┘
                │                              │
                ▼                              ▼
         [User Interface]              [LangSmith Tracing Engine]

```

```

### 2. Folder Structure


```

project-root/
├── .github/
│   └── workflows/
│       └── deploy.yml          # Automated CI/CD Git Sync Pipeline
├── src/
│   ├── data_processor.py       # Raw to Processed ETL logic
│   ├── ingestion.py            # Chunking, vector indexing & Embedding generation
│   ├── pipeline.py             # RAG Orchestration (Nesting Retriever + Generator)
│   └── evaluation_utils.py     # RAGAS Benchmark Evaluators & Native Judge Factories
├── backend/
│   └── main.py                 # FastAPI REST Engine & Background Worker tasks
├── frontend/
│   └── app.py                  # Streamlit Viewport Dashboard UX & Auto-spawner
├── data/
│   ├── raw/                    # Source technical manuscripts
│   ├── processed/              # Evaluation CSV reports
│   └── uploads/                # Dynamic user ingestion folder
├── chroma_db/                  # Persistent Vector database snapshot local disk
├── Dockerfile                  # Production Monolith Supervisor Image blueprint
├── docker-compose.yml          # Local container development coordinator
├── metrics.json                # Live-updating system performance benchmark storage
└── requirements.txt            # Explicitly pinned application dependencies

```

---

## Database Design

The system implements a persistent document vector storage design mapping locally to disk under the `chroma_db/` volume configuration. Raw text manuscripts (`.pdf`, `.txt`, `.md`) undergo automated ETL processing:
1. Parsing via specialized loaders (`PyPDFLoader`, `TextLoader`, `UnstructuredMarkdownLoader`).
2. Chunking via `RecursiveCharacterTextSplitter` (chunk size: $1000$, overlap: $200$).
3. Dense vector seeding through `models/gemini-embedding-001`.
4. Persistence inside `ChromaDB` with dynamic path anchoring to prevent container directory read/write locks.

---

## Installation

### Clone the Repository

```bash
git clone [https://github.com/Vivekbiju0/RAG-Document-Analyzer.git](https://github.com/Vivekbiju0/RAG-Document-Analyzer.git)
cd RAG-Document-Analyzer

```

### Environment Configuration

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=RAG-Document-Analyzer

```

### Run Local Container Stack

To spin up the multi-process stack using Docker Compose:

```bash
docker-compose up --build

```

---

## Usage

1. Open the UI via local port settings or navigate to the [Hugging Face Live Demo](https://huggingface.co/spaces/Vivekbiju0/RAG-Document-Analyzer).
2. Upload a technical manuscript or select an existing document inside the dynamic ingestion panel in the Control Center.
3. Submit queries through the interactive chat interface to inspect dual-stage context generation and evidence snippets.
4. Click **Run System Benchmark** in the sidebar to trigger background RAGAS evaluation runs.
5. Check the [LangSmith Dashboard](https://aws.smith.langchain.com/o/a01e2752-b31e-4cf6-b828-f7a2634b5944/projects/p/defd6317-d455-4f33-847a-b3ff3387a35b) to inspect real-time execution spans and latency graphs.

---

## Screenshots

*Live production interface tracking system health metrics, benchmark scores, and rendering evidence context spans.*

---

## Deployment

### Automated CI/CD Pipeline (GitHub Actions)

The repository automatically pushes system builds directly into production via `.github/workflows/deploy.yml` whenever code reaches the `main` branch.

To clear cache data or reset tracking dependencies locally, safely spin down active configurations:

```bash
docker-compose down

```

---

## Future Improvements

* Add comprehensive unit & integration testing suites (PyTest) for individual ETL steps.
* Implement automated real-time latency alerts for contextual fallback loops.
* Add interactive RAGAS score trend graphs directly inside the Streamlit Control Center.

---

## Credits

**Developer:** Vivek Biju

**GitHub:** [https://github.com/vivekbiju](https://www.google.com/search?q=https://github.com/vivekbiju)

---

## License

This project is licensed under the [MIT License](https://www.google.com/search?q=./LICENSE).

```

```
