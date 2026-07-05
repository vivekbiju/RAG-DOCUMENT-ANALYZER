import os
import time
import json
from datetime import datetime

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Initialize environment variables immediately
load_dotenv()

from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from google import genai

from langchain_core.tracers.context import collect_runs
from src.pipeline import GeminiRAG


def run_evaluation():
    # Initialize RAG system
    rag_system = GeminiRAG()
    
    # 1. Initialize the official Google GenAI Client
    google_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    # 2. Build the native judge LLM and Embeddings via factories
    judge_llm = llm_factory(
        model="gemini-2.5-flash",
        provider="google",
        client=google_client
    )

    judge_embeddings = embedding_factory(
        model="text-embedding-004",
        provider="google",
        client=google_client
    )

    test_set = [
        {
            "question": "What is the core benefit of the Transformer over RNNs?",
            "ground_truth": "The Transformer allows for significantly more parallelization and requires less time to train compared to recurrent models."
        },
        {
            "question": "Explain the Scaled Dot-Product Attention formula.",
            "ground_truth": "It computes the dot products of the query with all keys, divides by the square root of the key dimension, and applies a softmax function to the values."
        }
    ]

    results = []
    
    for i, item in enumerate(test_set):
        print(f"Processing {i+1}/{len(test_set)}: {item['question']}")
        
        # Retrieval
        relevant_docs = rag_system.retrieve_and_rerank(item['question'])
        contexts = [doc.page_content for doc in relevant_docs]

        # Generation
        answer = rag_system.generate(item['question'], relevant_docs)

        results.append({
            "question": item['question'],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item['ground_truth']
        })
        
        if i < len(test_set) - 1:
            print("Pausing to respect rate limits...")
            time.sleep(15) 

    dataset = Dataset.from_list(results)
    print("\n--- Running RAGAS Evaluation ---")
    
    with collect_runs() as cb:
        try:
            # 3. Instantiate standard metrics without arguments
            metrics = [
                Faithfulness(), 
                AnswerRelevancy(), 
                ContextPrecision(), 
                ContextRecall()
            ]

            # 4. Pass the global LLM and Embeddings directly into evaluate()
            score = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_embeddings
            )
            
            run_id = cb.traced_runs[0].id if cb.traced_runs else None
            print(f"Traced to LangSmith! Run ID: {run_id}")

            df = score.to_pandas()

            # --- CALCULATE MEAN SCORES FOR UI ---
            summary_metrics = {
                "faithfulness": float(df['faithfulness'].mean()),
                "relevancy": float(df['answer_relevancy'].mean()),
                "precision": float(df['context_precision'].mean()),
                "total_tests": len(test_set),
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M")
            }

            # Save to metrics.json for Streamlit to read
            with open("metrics.json", "w") as f:
                json.dump(summary_metrics, f, indent=4)

            # Save detailed CSV
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv("data/processed/evaluation_report.csv", index=False)

            print("\n✅ Evaluation Complete!")
            print(f"Mean Faithfulness: {summary_metrics['faithfulness']:.2%}")
            print(f"Mean Relevancy: {summary_metrics['relevancy']:.2%}")
            print("Results exported to metrics.json and evaluation_report.csv")

        except Exception as e:
            print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    run_evaluation()