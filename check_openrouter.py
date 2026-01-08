"""
OpenRouter API Checker with Multiple Fallback Models
Tests various free and paid models to find what works
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


# List of models to try (from free to paid)
MODELS_TO_TEST = [
    # FREE MODELS (Good for testing)
    {
        "name": "google/gemini-flash-1.5-8b",
        "provider": "Google",
        "cost": "Free",
        "description": "Gemini Flash 1.5 8B - Fast and free"
    },
    {
        "name": "google/gemini-2.0-flash-exp:free",
        "provider": "Google", 
        "cost": "Free",
        "description": "Gemini 2.0 Flash Experimental - Latest experimental"
    },
    {
        "name": "meta-llama/llama-3.2-3b-instruct:free",
        "provider": "Meta",
        "cost": "Free",
        "description": "Llama 3.2 3B - Small but capable"
    },
    {
        "name": "microsoft/phi-3-mini-128k-instruct:free",
        "provider": "Microsoft",
        "cost": "Free",
        "description": "Phi-3 Mini - Good for code"
    },
    {
        "name": "qwen/qwen-2-7b-instruct:free",
        "provider": "Alibaba",
        "cost": "Free",
        "description": "Qwen 2 7B - Strong multilingual"
    },
    
    # PAID BUT CHEAP (Recommended for production)
    {
        "name": "google/gemini-flash-1.5",
        "provider": "Google",
        "cost": "$0.075/$0.30 per 1M tokens",
        "description": "Gemini Flash 1.5 - Best balance"
    },
    {
        "name": "anthropic/claude-3-haiku",
        "provider": "Anthropic",
        "cost": "$0.25/$1.25 per 1M tokens",
        "description": "Claude 3 Haiku - Fast and cheap"
    },
    {
        "name": "openai/gpt-4o-mini",
        "provider": "OpenAI",
        "cost": "$0.15/$0.60 per 1M tokens",
        "description": "GPT-4o Mini - Great for code"
    },
    {
        "name": "meta-llama/llama-3.1-8b-instruct",
        "provider": "Meta",
        "cost": "$0.06/$0.06 per 1M tokens",
        "description": "Llama 3.1 8B - Very cheap"
    },
]


def test_model(model_config: dict, api_key: str) -> tuple[bool, str]:
    """
    Test a single model
    
    Returns:
        (success: bool, response: str)
    """
    try:
        llm = ChatOpenAI(
            model=model_config["name"],
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=50  # Keep it short for testing
        )
        
        response = llm.invoke("Say 'API working!' in 3 words.")
        return True, response.content
        
    except Exception as e:
        error_msg = str(e)
        
        # Simplify error messages
        if "404" in error_msg:
            return False, "Model not found (404)"
        elif "401" in error_msg:
            return False, "Invalid API key (401)"
        elif "429" in error_msg:
            return False, "Rate limit exceeded (429)"
        elif "quota" in error_msg.lower():
            return False, "Quota exceeded"
        else:
            return False, f"Error: {error_msg[:50]}..."


def main():
    print("=" * 70)
    print("🔍 OpenRouter API Checker - Testing Multiple Models")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("\n❌ OPENROUTER_API_KEY not found in .env file")
        print("\nSetup instructions:")
        print("1. Go to: https://openrouter.ai/keys")
        print("2. Create an API key")
        print("3. Add to .env file: OPENROUTER_API_KEY=sk-or-v1-...")
        return
    
    print(f"\n✅ API key found: {api_key[:20]}...")
    print(f"\n📋 Testing {len(MODELS_TO_TEST)} models...\n")
    
    working_models = []
    failed_models = []
    
    # Test each model
    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"[{i}/{len(MODELS_TO_TEST)}] Testing: {model['name']}")
        print(f"    Provider: {model['provider']} | Cost: {model['cost']}")
        
        success, result = test_model(model, api_key)
        
        if success:
            print(f"    ✅ SUCCESS! Response: {result}")
            working_models.append(model)
        else:
            print(f"    ❌ FAILED: {result}")
            failed_models.append((model, result))
        
        print()
    
    # Summary
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if working_models:
        print(f"\n✅ {len(working_models)} model(s) working:\n")
        
        for model in working_models:
            print(f"  • {model['name']}")
            print(f"    {model['description']}")
            print(f"    Cost: {model['cost']}\n")
        
        # Recommend best model
        print("🎯 RECOMMENDED FOR YOUR PROJECT:")
        
        # Prefer free models for development
        free_models = [m for m in working_models if m['cost'] == 'Free']
        if free_models:
            best = free_models[0]
            print(f"\n  Model: {best['name']}")
            print(f"  Why: {best['description']} + It's FREE!")
        else:
            best = working_models[0]
            print(f"\n  Model: {best['name']}")
            print(f"  Why: {best['description']}")
            print(f"  Cost: {best['cost']}")
        
        print(f"\n📝 Add this to your swarm configuration:")
        print(f'   model_name = "{best["name"]}"')
        
    else:
        print("\n❌ No models working!")
        print("\nPossible issues:")
        print("  • Invalid API key")
        print("  • No credits on account")
        print("  • Network issues")
        print("\nTry:")
        print("  1. Check your balance: https://openrouter.ai/credits")
        print("  2. Add credits if needed")
        print("  3. Verify API key is active")
    
    if failed_models:
        print(f"\n⚠️ {len(failed_models)} model(s) failed:")
        for model, error in failed_models[:5]:  # Show first 5
            print(f"  • {model['name']}: {error}")


if __name__ == "__main__":
    main()