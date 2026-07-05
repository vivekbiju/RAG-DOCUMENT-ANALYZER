import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables (including GOOGLE_API_KEY)
load_dotenv()

class GeminiRAG:
    def __init__(self, db_dir="./chroma_db"):
        # 1. Initialize Embeddings (Auto-pulls API Key from env)
        #self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-1.5-flash")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # 2. Load the Vector Store
        self.vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=self.embeddings
        )

        # 3. Initialize the LLM (Using a standard stable version)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.3
        )
    def retrieve(self, query, k=3):
        """Finds the most relevant technical snippets."""
        print(f"\n[RETRIEVAL] Searching for: '{query}'")
        return self.vectorstore.similarity_search(query, k=k)

    def generate(self, query, contexts):
        """Constructs the prompt and gets an answer from Gemini."""
        context_text = "\n\n---\n\n".join([doc.page_content for doc in contexts])
        prompt = f"""You are a technical AI expert. Answer the question using ONLY the provided research context.
        If the answer is not present, state that you do not have enough information.

        CONTEXT:
        {context_text}

        QUESTION: {query}

        ANSWER:"""

        print(f"[GENERATION] Consulting Gemini...")
        response = self.llm.invoke(prompt)

        # FIX: Check if response is a list/dict and extract the 'text' field
        # 1. Check if the response content is a list (like in your screenshot)
        if isinstance(response.content, list):
        # Extract only the 'text' part from the first element
            first_part = response.content[0]
            if isinstance(first_part, dict) and 'text' in first_part:
                return first_part['text']
            else:
                return str(first_part)
            
        # 2. Check if it is a standard string
        elif isinstance(response.content, str):
            return response.content
        return "No valid content found in response."
        # Fallback for older SDK versions
        return getattr(response, 'content', str(response))

    def query_system(self, question):
        """The main orchestration flow."""
        relevant_docs = self.retrieve(question)
        answer = self.generate(question, relevant_docs)
        return answer

if __name__ == "__main__":
    # Test the system with a question about the 'Attention' paper
    rag = GeminiRAG()
    test_query = "What are the two main components of the Transformer architecture?"

    # Based on our doc, it should mention the Encoder and Decoder stacks.
    result = rag.query_system(test_query)
    print("\n" + "="*50)
    print(f"QUESTION: {test_query}")
    print(f"ANSWER:\n{result}")
    print("="*50)