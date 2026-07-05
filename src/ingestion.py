import os
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables once at the module level
load_dotenv()


class IngestionPipeline:
    def __init__(self, db_dir="./chroma_db", model_name="models/gemini-embedding-001"):
        self.db_dir = db_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
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
        Processes a single file: Loads, Chunks, and Adds to the ChromaDB.
        """
        # 1. Load the document
        docs = self.load_document(file_path)
        
        # 2. Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks from {file_path}")

        # 3. Add to Vector Store
        # We initialize the Chroma object first.
        # If the directory exists, it loads it; if not, it prepares to create it.
        print(f"Updating vector store at {self.db_dir}...")
        
        vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings
        )
        
        # Add the new chunks to the existing collection
        vectorstore.add_documents(chunks)
        
        print(f"Successfully added {file_path} to the knowledge base.")
        return vectorstore


if __name__ == "__main__":
    # Local testing logic
    test_file = os.path.join("data", "processed_data", "docs_clean.txt")
   
    if os.path.exists(test_file):
        pipeline = IngestionPipeline()
        pipeline.run(test_file)
    else:
        print(f"Error: {test_file} not found.")