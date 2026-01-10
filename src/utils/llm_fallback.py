"""
LLM Fallback Mechanism for OpenRouter Free Models.
Automatically tries multiple models if one fails.
"""

import os
import time
from typing import List, Optional, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk

load_dotenv()

# List of free models that support tool calling
FREE_MODELS = [
    "mistralai/devstral-2512:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


class NonStreamingChatOpenAI(ChatOpenAI):
    """Wrapper that forces non-streaming behavior for OpenRouter compatibility."""
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # Override stream to use non-streaming invoke instead
        kwargs.pop("stream", None)
        kwargs["stream"] = False
        
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                additional_kwargs=message.additional_kwargs,
                id=message.id if hasattr(message, 'id') else None,
            )
        )
        yield chunk
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("stream", None)
        kwargs["stream"] = False
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class FallbackLLM:
    """
    LLM wrapper with automatic fallback to alternative models.
    Tries each model in the list until one succeeds.
    """
    
    def __init__(self, api_key: str, models: List[str] = None, temperature: float = 0):
        self.api_key = api_key
        self.models = models or FREE_MODELS
        self.temperature = temperature
        self.current_model_index = 0
        self._llm = None
        self._create_llm()
    
    def _create_llm(self, model_index: int = 0) -> NonStreamingChatOpenAI:
        """Create an LLM instance for the specified model index."""
        model = self.models[model_index]
        print(f"🔄 Initializing LLM with model: {model}")
        self._llm = NonStreamingChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=self.temperature,
            base_url="https://openrouter.ai/api/v1",
            streaming=False,
            model_kwargs={},
        )
        self.current_model_index = model_index
        return self._llm
    
    def get_llm(self) -> NonStreamingChatOpenAI:
        """Get the current LLM instance."""
        return self._llm
    
    def try_next_model(self) -> Optional[NonStreamingChatOpenAI]:
        """Switch to the next model in the fallback list."""
        next_index = self.current_model_index + 1
        if next_index < len(self.models):
            print(f"⚠️ Switching to fallback model...")
            return self._create_llm(next_index)
        else:
            print("❌ All fallback models exhausted!")
            return None
    
    def reset(self):
        """Reset to the first model."""
        self._create_llm(0)


def create_fallback_llm(api_key: str, models: List[str] = None) -> NonStreamingChatOpenAI:
    """
    Simple function to create an LLM with the first available model.
    For more advanced fallback, use the FallbackLLM class.
    """
    models = models or FREE_MODELS
    return NonStreamingChatOpenAI(
        model=models[0],
        api_key=api_key,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        streaming=False,
        model_kwargs={},
    )


def invoke_with_fallback(agent_executor, input_dict: dict, models: List[str] = None, api_key: str = None, max_retries: int = 3):
    """
    Invoke an agent executor with automatic model fallback on failure.
    
    Args:
        agent_executor: The AgentExecutor to invoke
        input_dict: The input dictionary for the agent
        models: List of models to try
        api_key: OpenRouter API key
        max_retries: Maximum number of models to try
    
    Returns:
        The agent response or raises the last exception
    """
    models = models or FREE_MODELS
    last_exception = None
    
    for i, model in enumerate(models[:max_retries]):
        try:
            print(f"🔄 Attempting with model: {model}")
            response = agent_executor.invoke(input_dict, config={"callbacks": []})
            return response
        except Exception as e:
            error_str = str(e).lower()
            last_exception = e
            
            # Check if it's a retryable error
            if any(x in error_str for x in ["429", "rate limit", "streaming", "tools are not supported"]):
                print(f"⚠️ Model {model} failed: {str(e)[:100]}...")
                if i < max_retries - 1:
                    print(f"⏳ Waiting 5 seconds before trying next model...")
                    time.sleep(5)
                continue
            else:
                # Non-retryable error, raise immediately
                raise e
    
    # All retries failed
    raise last_exception
