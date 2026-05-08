# tests/sync_langsmith.py
import os
from langsmith import Client
from tests.custom_eval import test_set  
from dotenv import load_dotenv

# 1. Load variables from .env file
load_dotenv()

# 2. Initialize client (it automatically reads from your environment)
client = Client()
dataset_name = "RAG_Research_Papers"

def sync():
    if not client.has_dataset(dataset_name=dataset_name):
        print(f"Creating dataset: {dataset_name}")
        client.create_dataset(dataset_name=dataset_name)
            
        for item in test_set:
            client.create_example(
                inputs={"question": item["question"]},
                outputs={"ground_truth": item["ground_truth"]},
                dataset_name=dataset_name
            )
        print("✅ Sync complete!")
    else:
        print(f"Dataset '{dataset_name}' already exists in LangSmith.")

if __name__ == "__main__":
    sync()