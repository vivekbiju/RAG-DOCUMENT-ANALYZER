---
title: RAG Document Analyzer
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🔬 Advanced RAG Research Assistant: Transformer Architecture

A production-ready, enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to query complex technical research papers. This project demonstrates an advanced **two-stage hybrid retrieval pipeline** leveraging **Google Gemini 2.5 Flash** and a **FlashRank Cross-Encoder Reranker**, coupled with robust multi-container microservices, enterprise LLMOps logging, and automated CI/CD.

## 🔗 Live Application & Observability
* **Live Production App:** [Insert your Hugging Face Space Link Here]
* **Observability Dashboard:** [Insert your LangSmith Project Link Here]

---

## 📸 System Interface
Below is a live look at the production interface tracking and rendering advanced context-retrieval spans:

![System Interface Screenshot](image_5fd246.png)

---

## 🏗️ Production Architecture & Engineering Evolution
The application has transitioned from a loose local script collection into a hardened, production-ready **Monolith-in-a-Box** architecture optimized for free-tier cloud constraints (like Hugging Face Spaces).

```text
       ┌─────────────────────────────────────────────────────────┐
       │                 Hugging Face Spaces (Pod)               │
       │                                                         │
       │     ┌─────────────────────────────────────────────┐     │
       │     │            Supervisor Process Manager       │     │
       │     └──────┬──────────────────────────────┬───────┘     │
       │            │                              │             │
       │            ▼                              ▼             │
       │ ┌──────────────────────┐      ┌───────────────────────┐ │
       │ │   Streamlit Frontend │      │     FastAPI Backend   │ │
       │ │     (Port 7860)      │      │       (Port 8000)     │ │
       │ └──────────┬───────────┘      └───────────┬───────────┘ │
       └────────────┼──────────────────────────────┼─────────────┘
                    │                              │
                    ▼                              ▼
             [User Interface]              [LangSmith Tracing Engine]

```

### 1. Architectural Re-engineering (Separation of Concerns)

* **Testing isolation:** Evaluation utilities were refactored completely out of the local `/tests` layout and moved into `src/evaluation_utils.py`. This ensures production container images remain clean, lightweight, and completely decoupled from testing dependencies, preventing `ModuleNotFoundError` during cloud compilation.
* **Stateful Volume Binding:** Configured specific Docker storage mappings for directory persistence (`chroma_db/`, `data/`, and `metrics.json`), allowing seamless hot-reloading across the front-to-back architecture.

### 2. Multi-Process Supervisor Orchestration

To deploy on free cloud architectures that restrict environments to a single open ingress port (`7860`), the system utilizes an optimized Linux **Supervisor multi-process manager**. This lightweight orchestration layer boots and manages both the background FastAPI application server and the interactive Streamlit user viewport concurrently inside a unified runtime layer.

### 3. Deep LLMOps Observability (LangSmith Integration)

The system moves past disjointed logging by executing parent context nesting. Using the native `@traceable` decorator framework linked directly to custom environment hooks, individual spans (`RAG_Retriever` and `RAG_Generator`) are collected, grouped, and streamed out as a singular synchronized tree trace execution graph.

---

## 🛠️ The Tech Stack

* **Core Core LLM:** Google Gemini 2.5 Flash (Optimized for ultra-low latency generation)
* **Embeddings:** Google Generative AI Engine (`models/gemini-embedding-001`)
* **Framework Orchestration:** LangChain / LangChain-Chroma Expression syntax
* **Vector Infrastructure:** ChromaDB (Persistent Vector Storage Engine)
* **Reranking Engine:** FlashRank (Lightweight, local Cross-Encoder framework)
* **Process Monitor:** Supervisor daemon (Linux process management)
* **Deployment System:** Docker / GitHub Actions CI/CD pipeline

---

## 📁 Project Directory Layout

```text
project-root/
├── .github/
│   └── workflows/
│       └── deploy.yml          # Automated CI/CD Git Sync Pipeline
├── src/
│   ├── data_processor.py       # Raw to Processed ETL logic
│   ├── ingestion.py            # Chunking, vector indexing & Embedding generation
│   ├── pipeline.py             # RAG Orchestration (Nesting Retriever + Generator)
│   └── evaluation_utils.py     # Relocated RAGAS Benchmark Evaluators
├── backend/
│   └── main.py                 # FastAPI REST Engine & Background Worker tasks
├── frontend/
│   └── app.py                  # Streamlit Viewport Dashboard UX
├── data/
│   ├── raw/                    # Source technical manuscripts
│   └── uploads/                # Dynamic user ingestion folder
├── chroma_db/                  # Persistent Vector database snapshot local disk
├── Dockerfile                  # Production Monolith Supervisor Image blueprint
├── docker-compose.yml          # Local container development coordinator
├── metrics.json                # Live-updating system performance benchmark storage
└── requirements.txt            # Explicitly pinned application dependencies

```

---

## 📈 Engineering Rigor & Features

* **Two-Stage Retrieval (Recall vs Precision):** Fetches the top 10 most candidate document nodes via vector similarity (Stage 1), then passes them through a localized Cross-Encoder model to bubble up the most contextually relevant information (Stage 2). This eliminates "lost-in-the-middle" LLM context degradation.
* **Non-Blocking Background Metrics:** The `/run-benchmark` pipeline offloads compute-heavy evaluations into an asynchronous `BackgroundTasks` queue. This permits immediate frontend server response while deep model evaluations continue running under the hood.
* **Secure Environment Architecture:** Zero-hardcoded credentials. Sensitive access variables are routed strictly out of ephemeral container memory configurations, keeping internal cloud keys protected.

---

## 🚀 Automated CI/CD Deployment (GitHub Actions)

The system relies on an automated continuous delivery channel. Every change pushed to the repository triggers the `.github/workflows/deploy.yml` pipeline, instantly pushing code updates over to the production environment:

```yaml
name: Sync to Hugging Face Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Push to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: git push --force https://Vivekbiju0:$HF_TOKEN@huggingface.co/spaces/Vivekbiju0/RAG-Document-Analyzer.git main:main

```

### Locally Rebuilding the App Stack

To verify changes or test dependencies locally before committing code to GitHub, simply spin down the active containers and trigger an explicit build update:

```bash
docker-compose down
docker-compose up --build

```

```
