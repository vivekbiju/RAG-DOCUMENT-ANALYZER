import os
import sys
import shutil
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables once at the module level
load_dotenv()

class IngestionPipeline:
    def __init__(self, db_dir=None, model_name="text-embedding-004"):
        """
        Initializes the Ingestion Pipeline.
        Anchors paths dynamically to avoid OS-level directory locks in production containers.
        """
        base_path = Path(__file__).resolve().parent.parent
        self.db_dir = str(base_path / "chroma_db") if db_dir is None else db_dir
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path):
        """Loads document based on file extension."""
        print(f"--- Loading: {file_path} ---")
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path}")
        
        return loader.load()

    def run(self, file_path):
        """
        Processes a single file: Loads, Chunks, and Adds to the ChromaDB collection.
        """
        # 1. Load the document
        docs = self.load_document(file_path)
        
        # 2. Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks from {file_path}")

        # 3. Add to Vector Store
        print(f"Updating vector store at {self.db_dir}...")
        
        # Native integration using the strictly typed LangChain-Chroma vector wrapper
        vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )
        
        # Add the new chunks to the existing collection
        vectorstore.add_documents(chunks)
        
        print(f"Successfully added {file_path} to the knowledge base.")
        return vectorstore


if __name__ == "__main__":
    # Local testing entry path handler
    base_path = Path(__file__).resolve().parent.parent
    test_file = base_path / "data" / "processed_data" / "docs_clean.txt"
   
    if test_file.exists():
        pipeline = IngestionPipeline()
        pipeline.run(str(test_file))
    else:
        print(f"Error: {test_file} not found.")