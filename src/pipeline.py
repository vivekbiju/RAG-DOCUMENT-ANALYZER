import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langsmith import traceable

# ROBUST IMPORT STRATEGY
try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        ContextualCompressionRetriever = None

try:
    from langchain_community.document_compressors.flashrank import FlashrankRerank
except ImportError:
    FlashrankRerank = None

load_dotenv()

class GeminiRAG:
    def __init__(self, db_dir="./chroma_db"):
        # 1. Initialize Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # 2. Load Vector Store
        self.vectorstore = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        # 3. Initialize Reranker (Two-Stage Retrieval)
        if FlashrankRerank and ContextualCompressionRetriever:
            try:
                compressor = FlashrankRerank()
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                print("System initialized with FlashRank Reranker.")
            except Exception as e:
                print(f"Reranker init failed: {e}. Falling back to base.")
                self.retriever = base_retriever
        else:
            self.retriever = base_retriever
            print("System initialized with Base Retriever.")

        # 4. Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    @traceable(name="RAG_Retriever", run_type="retriever") 
    def retrieve_and_rerank(self, question):
        """Used by Streamlit to get docs for the 'Evidence' expander."""
        print(f"[RETRIEVAL] Fetching candidates for: '{question}'")
        return self.retriever.invoke(question)

    @traceable(name="RAG_Generator", run_type="llm") 
    def generate(self, question, relevant_docs):
        """Core generation logic using a technical persona."""
        context_text = "\n\n---\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""You are a technical AI expert. Answer the question using ONLY the provided context.
        If the answer is not in the context, say you don't have enough information.

        CONTEXT:
        {context_text}

        QUESTION: {question}

        ANSWER:"""

        print(f"[GENERATION] Consulting Gemini...")
        response = self.llm.invoke(prompt)
        
        # Robust content parsing
        content = response.content
        if isinstance(content, list):
            return " ".join([part['text'] if isinstance(part, dict) and 'text' in part else str(part) for part in content])
        return content

    def query_system(self, question):
        """Standard orchestration for the terminal/CLI mode."""
        docs = self.retrieve_and_rerank(question)
        return self.generate(question, docs)

if __name__ == "__main__":
    # Standard CLI loop for local testing
    rag = GeminiRAG()
    print("\nREADY: Ask about 'Attention Is All You Need'")
    
    while True:
        user_query = input("\n[USER]: ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            break
        if not user_query.strip():
            continue
            
        result = rag.query_system(user_query)
        print(f"\n[AI]: {result}")