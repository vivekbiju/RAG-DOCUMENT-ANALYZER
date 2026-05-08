import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
print(f"API Key found: {os.environ.get('LANGCHAIN_API_KEY')[:10]}...") 

client = Client()
try:
    datasets = list(client.list_datasets(limit=1))
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")