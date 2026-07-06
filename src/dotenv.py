import os
from dotenv import load_dotenv

load_dotenv()

# Check for GEMINI_API_KEY first; if not found, fall back to GOOGLE_API_KEY
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("⚠️ WARNING: Neither GEMINI_API_KEY nor GOOGLE_API_KEY was found!")