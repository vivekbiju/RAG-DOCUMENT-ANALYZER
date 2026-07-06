import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
from langsmith import traceable

# ROBUST PROCESS RETRIEVER IMPORT STRATEGY
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
    def __init__(self, db_dir=None):
        """
        Initializes the retrieval and generation pipeline.
        Anchors database paths dynamically to avoid directory read/write locks in container clusters.
        """
        base_path = Path(__file__).resolve().parent.parent
        self.db_dir = str(base_path / "chroma_db") if db_dir is None else db_dir

        # 1. FIXED: Corrected class reference and unified embedding model tracking
        self.embeddings = GoogleGenAIEmbeddings(model="text-embedding-004")
        
        # 2. Load Vector Store Instance cleanly
        self.vectorstore = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # 3. Initialize Two-Stage FlashRank Reranker if dependencies are met
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
            
        # 4. Initialize Core LLM Generation Engine
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    @traceable(name="RAG_Retriever", run_type="retriever")
    def retrieve_and_rerank(self, question):
        """Used by Streamlit to fetch documents for the 'Evidence' expander dashboard layer."""
        print(f"[RETRIEVAL] Fetching candidates for: '{question}'")
        try:
            return self.retriever.invoke(question)
        except Exception as e:
            print(f"⚠️ Retrieval layer warning (index might be completely empty): {e}")
            return []

    @traceable(name="RAG_Generator", run_type="llm")
    def generate(self, question, relevant_docs):
        """Core generation logic with defensive empty-state and multi-type payload parsing blocks."""
        # 1. Handle empty initialization indexes gracefully
        if not relevant_docs:
            return "The knowledge base is currently empty. Please drop or upload research documents via the sidebar manager to get started!"

        context_text = "\n\n---\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""You are a technical AI expert. Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't have enough information.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

        print("[GENERATION] Consulting Gemini API Engine...")
        try:
            response = self.llm.invoke(prompt)
            
            # 2. Extract contents safely based on primitive/object response payload variations
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            if isinstance(content, list):
                parsed_parts = []
                for part in content:
                    if isinstance(part, dict) and 'text' in part:
                        parsed_parts.append(part['text'])
                    elif isinstance(part, dict) and 'content' in part:
                        parsed_parts.append(part['content'])
                    else:
                        parsed_parts.append(str(part))
                return " ".join(parsed_parts)
                
            return str(content)
            
        except Exception as e:
            print(f"❌ Generation workflow run execution fault: {e}")
            return f"An error occurred during response generation processing: {str(e)}"

    @traceable(name="RAG_Pipeline", run_type="chain")
    def query_system(self, question):
        """Standard orchestration loop mapping out CLI access and main trace spans."""
        docs = self.retrieve_and_rerank(question)
        answer = self.generate(question, docs)
        return answer


if __name__ == "__main__":
    # Standard terminal execution loop for staging / smoke testing
    rag = GeminiRAG()
    print("\nREADY: Ask about target research documents...")
    
    while True:
        user_query = input("\n[USER]: ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            break
        if not user_query.strip():
            continue
            
        result = rag.query_system(user_query)
        print(f"\n[AI]: {result}")