# test_gemini.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

load_dotenv()
GEMINI_API_KEY = "AIzaSyCTNWic7m_rbMnNIXCD0BnKj4KVbQhAXZE"

if not GEMINI_API_KEY:
    print("API Key not found in .env or environment!")
    exit()

print("Attempting to configure Gemini...")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use a model known to be generally available, like gemini-1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini configured. Attempting API call...")
    start_time = time.time()
    # Simple prompt
    response = model.generate_content("List 3 synonyms for 'happy'")
    end_time = time.time()
    print(f"API call successful! Took {end_time - start_time:.2f} seconds.")
    print("Response text:")
    print(response.text)
except Exception as e:
    print(f"ERROR during Gemini configuration or API call: {e}")
    import traceback
    traceback.print_exc()

print("Test script finished.")