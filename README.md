# 🤖 Advanced RAG Research Assistant: 'Attention Is All You Need'

A professional-grade Retrieval-Augmented Generation (RAG) system built to query complex technical research papers. This project demonstrates an advanced two-stage retrieval pipeline using **Google Gemini 1.5 Flash** and **FlashRank Reranking**.

## 🚀 Live Demo
[Insert your Streamlit Link Here]

---

## 🏗️ System Architecture
The system follows a modular "Data Lake" philosophy to ensure scalability and data governance:

1.  **Ingestion Pipeline**: Processes the raw "Attention Is All You Need" PDF/TXT.
2.  **Vector Store**: Generates embeddings using `gemini-embedding-001` and stores them in a persistent ChromaDB.
3.  **Two-Stage Retrieval**:
    *   **Stage 1 (Recall)**: Fetches top 10 candidate chunks via vector similarity search.
    *   **Stage 2 (Precision)**: Uses a **FlashRank Reranker** to re-sort candidates, ensuring the most technically relevant context is sent to the LLM.
4.  **Generation**: Gemini 1.5 Flash generates a grounded response using a "Technical Expert" persona.

---

## 🛠️ Tech Stack
*   **LLM**: Google Gemini 1.5 Flash (Latest)
*   **Embeddings**: Google Generative AI (`models/gemini-embedding-001`)
*   **Orchestration**: LangChain
*   **Vector Database**: ChromaDB
*   **Reranker**: FlashRank (Lightweight Cross-Encoder)
*   **UI/Deployment**: Streamlit

---

## 📈 Key Features & Engineering Rigor
*   **Contextual Reranking**: Solves the "lost in the middle" problem of standard vector search by validating chunk relevance before generation.
*   **Robust Data Processing**: Implements a `raw` vs `processed` data workflow, standardizing text from research papers for better chunking.
*   **Defense-in-Depth Handling**: Includes robust error handling for API failures and fallback logic if the Reranker is unavailable.
*   **LLM-as-a-Judge Evaluation**: Features a custom evaluation framework using Gemini 1.5 Pro to grade the system on **Faithfulness** and **Accuracy** against ground-truth paper data[cite: 1].

---

## 📁 Project Structure
```text
project-root/
├── src/
│   ├── data_processor.py   # Raw to Processed ETL logic
│   ├── ingestion.py        # Vector indexing & Embedding logic
│   ├── pipeline.py         # Main RAG orchestration (Retriever + Generator)
│   └── custom_eval.py      # LLM-as-a-Judge evaluation scripts
├── data/
│   ├── raw/                # Original research paper (docs.txt)
│   └── processed/          # Cleaned text & Evaluation reports
├── chroma_db/              # Persistent vector storage
├── app.py                  # Streamlit Web Interface
└── requirements.txt        # Pinned dependencies