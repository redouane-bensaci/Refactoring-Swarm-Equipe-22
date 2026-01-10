"""Test script to check rate limits for all 3 OpenRouter API keys."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEYS = {
    "OPENROUTER_API_KEY (auditor)": os.getenv("OPENROUTER_API_KEY"),
    "OPENROUTER_API_KEY2 (fixer)": os.getenv("OPENROUTER_API_KEY2"),
    "OPENROUTER_API_KEY3 (judge)": os.getenv("OPENROUTER_API_KEY3"),
}

MODEL = "meta-llama/llama-3.3-70b-instruct:free"

def test_api_key(name: str, api_key: str):
    """Test a single API key with a hello request."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    
    if not api_key:
        print("❌ API key not set!")
        return
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
            stream=False,
        )
        
        print(f"✅ SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

def main():
    print("Testing OpenRouter API Keys for rate limits...")
    print(f"Model: {MODEL}")
    
    for name, key in API_KEYS.items():
        test_api_key(name, key)
    
    print(f"\n{'='*50}")
    print("Test complete!")

if __name__ == "__main__":
    main()
