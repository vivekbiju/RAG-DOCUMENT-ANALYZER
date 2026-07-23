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


A production-ready, enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to query complex technical research papers. This project demonstrates an advanced two-stage hybrid retrieval pipeline leveraging Google Gemini 2.5 Flash and a FlashRank Cross-Encoder Reranker, coupled with robust multi-container microservices, enterprise LLMOps logging, and automated CI/CD.

Live Demo: https://huggingface.co/spaces/Vivekbiju0/RAG-Document-Analyzer  
Observability Dashboard: https://aws.smith.langchain.com/o/a01e2752-b31e-4cf6-b828-f7a2634b5944/projects/p/defd6317-d455-4f33-847a-b3ff3387a35b  
Production Repo: https://github.com/Vivekbiju0/RAG-Document-Analyzer

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Database Design](#database-design)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
- [License](#license)


## Overview

### Motivation
Standard RAG pipelines often suffer from "lost-in-the-middle" context degradation and heavy testing bloat in production environments. This project was engineered to solve these challenges by building a hardened, resource-optimized system that maintains high-precision retrieval on dense, technical datasets under free-tier cloud constraints.

### Objective
To deliver an isolated, dual-stage hybrid retrieval system capable of parsing technical manuscripts, tracking live context-retrieval spans through centralized LLMOps, and offloading heavy evaluation metrics without blocking user experience.

### Learning Outcomes
- Built a multi-process supervisor orchestration layer for single-ingress containers.
- Implemented a high-precision, two-stage hybrid retrieval framework (Recall vs Precision).
- Configured stateful volume binding and persistence configurations for vector storage.
- Established isolated evaluation frameworks to decouple production steps from development dependencies.
- Synchronized parent-child trace execution graphs using LangSmith integration.

## Features

- **Two-Stage Hybrid Retrieval:** Fetches top candidate nodes via vector similarity and refines them using a localized Cross-Encoder reranker.
- **Multi-Process Orchestration:** Runs a background FastAPI REST engine and an interactive Streamlit frontend concurrently via a Supervisor daemon.
- **Deep LLMOps Observability:** Centralized span tracking and execution graphs via native `@traceable` hooks.
- **Non-Blocking Background Metrics:** Asynchronous evaluation offloading via `BackgroundTasks` queue mapping.
- **Secure Environment Architecture:** Strict routing of keys through ephemeral container memory variables with zero hardcoded credentials.


## Tech Stack

### Frontend
- Streamlit Viewport Dashboard UX
- HTML5 / CSS3

### Backend
- FastAPI REST Engine & Background Workers
- LangChain / LangChain-Chroma Expression Syntax
- Supervisor Daemon (Linux Process Monitor)
- FlashRank (Lightweight local Cross-Encoder framework)

### Database
- ChromaDB (Persistent Vector Storage Engine)
- Google Generative AI Engine (`models/gemini-embedding-001`)

### Tools
- Google Gemini 2.5 Flash
- Docker & Docker Compose
- GitHub Actions CI/CD
- LangSmith Tracing Engine

## Architecture

### 1. Production Workflow

```
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


## Database Design

The system implements a persistent document vector storage design mapping locally to disk under the `chroma_db/` volume configuration. Raw text manuscripts undergo automated ETL processing, splitting data into optimized tokenized chunks before seeding vector spaces through the Google Generative AI embedding engine.


## Installation

### Clone the Repository

```bash
git clone [https://github.com/Vivekbiju0/RAG-Document-Analyzer.git](https://github.com/Vivekbiju0/RAG-Document-Analyzer.git)
cd RAG-Document-Analyzer

```

### Run Local Container Stack

To spin up the multi-process stack using Docker Compose:

```bash
docker-compose up --build

```

## Usage

1. Open the UI via local port settings or navigate to the Hugging Face live link.
2. Upload a technical manuscript or select an existing document inside the dynamic ingestion panel.
3. Submit queries through the interactive interface to inspect dual-stage context generation.
4. Check the LangSmith Observability link to verify performance logs and parent-child span execution steps.


## Screenshots

![alt text](image.png)

Below is a live look at the production interface tracking and rendering advanced context-retrieval spans:


## Deployment

### Automated CI/CD Pipeline (GitHub Actions)

The repository automatically pushes system builds directly into production via `.github/workflows/deploy.yml` whenever code reaches the `main` branch.

To clear cache data or reset tracking dependencies locally, safely spin down active configurations:

```bash
docker-compose down

```

## Future Improvements

* Add comprehensive unit & integration testing for individual ETL steps.
* Implement automated real-time alerts for context fallback loops.
* Add advanced tracking graphs directly inside the Streamlit user panel dashboard.


## Credits

Developer: Vivek Biju
GitHub: https://github.com/vivekbiju?tab=repositories

## License

This project is licensed under the [MIT License](./LICENSE).
