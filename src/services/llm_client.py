"""
LLM Client - Centralized model initialization
Supports both OpenRouter and Google AI
"""
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(
    model_name: str = "meta-llama/llama-3.2-3b-instruct:free",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    api_key: str = None
):
    """
    Get configured LLM instance
    
    Tries OpenRouter first, falls back to Google AI
    
    Args:
        model_name: Model identifier
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        
    Returns:
        Configured LLM instance
    """
    
    # Try OpenRouter first
    openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        print(f"🔧 Initializing OpenRouter: {model_name}")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    # Fallback to Google AI
    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        print(f"🔧 Initializing Google AI: gemini-2.5-flash")
        
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        )
    
    else:
        raise ValueError(
            "No API key found! Set either:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-... (in .env)\n"
            "  GOOGLE_API_KEY=AIza... (in .env)"
        )


def get_model_config(api_key: str = None):
    """
    Get current model configuration info
    
    Returns:
        dict with provider, model, and API key status
    """
    openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "provider": "OpenRouter",
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "api_key_set": True,
            "cost": "FREE"
        }
    elif os.getenv("GOOGLE_API_KEY"):
        return {
            "provider": "Google AI",
            "model": "gemini-2.5-flash",
            "api_key_set": True,
            "cost": "Free tier"
        }
    else:
        return {
            "provider": None,
            "model": None,
            "api_key_set": False,
            "cost": None
        }


# For testing
if __name__ == "__main__":
    config = get_model_config()
    
    print("=" * 60)
    print("🤖 LLM Configuration")
    print("=" * 60)
    
    if config["api_key_set"]:
        print(f"✅ Provider: {config['provider']}")
        print(f"✅ Model: {config['model']}")
        print(f"✅ Cost: {config['cost']}")
        
        # Test initialization
        try:
            llm = get_llm()
            response = llm.invoke("Say 'Hello' in 3 words")
            print(f"\n✅ Test response: {response.content}")
        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
    else:
        print("❌ No API key configured")
        print("\nSet one of:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        print("  GOOGLE_API_KEY=AIza...")