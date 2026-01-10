import os
import subprocess
from pathlib import Path
from typing import Dict, List, Union
from dotenv import load_dotenv

# Core LangChain 0.1.10 imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Setup Environment
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY2")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY2 environment variable is not set")

SANDBOX_ROOT = (Path(__file__).parent.parent.parent / "sandbox").resolve()

# 2. Initialize LLM - CRITICAL: No streaming
# Create a wrapper class to force non-streaming behavior
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessageChunk, AIMessage
from langchain_core.outputs import ChatGeneration

class NonStreamingChatOpenAI(ChatOpenAI):
    """Wrapper that forces non-streaming behavior for OpenRouter compatibility."""
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # Override stream to use non-streaming invoke instead
        # This prevents the "Tools not supported in streaming mode" error
        kwargs.pop("stream", None)
        kwargs["stream"] = False
        
        # Call _generate with stream explicitly disabled
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
        # Ensure stream is always False
        kwargs.pop("stream", None)
        kwargs["stream"] = False
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

llm = NonStreamingChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    streaming=False,
    model_kwargs={},
)

# 3. Define Tools
@tool
def read_file_content(file_path: str) -> str:
    """Reads the content of a file from the local disk.
    
    Args:
        file_path: Relative path from sandbox root (e.g., 'test3/add.py')
    """
    try:
        if file_path.startswith('sandbox/'):
            file_path = file_path.replace('sandbox/', '', 1)
            
        safe_path = (SANDBOX_ROOT / file_path).resolve()

        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"
        
        if not safe_path.exists():
            return f"File not found: {file_path}. Available files: {list_sandbox_files()}"

        if not safe_path.is_file():
            return f"Not a file: {file_path}"
        
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file_content(file_path: str, content: str) -> str:
    """Writes content to a file on the local disk. 
    Use this to save corrected or refactored code.
    
    Args:
        file_path: Relative path from sandbox root (e.g., 'test3/add.py')
        content: The corrected code content to write
    """
    try:
        if file_path.startswith('sandbox/'):
            file_path = file_path.replace('sandbox/', '', 1)
            
        safe_path = (SANDBOX_ROOT / file_path).resolve()

        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return f"✅ Successfully wrote corrected code to {file_path}"
    except Exception as e:
        return f"❌ Error writing to file: {str(e)}"

@tool

def read_directory(directory_path: str) -> Dict[str, Union[str, int, List[str]]]:
    """
    Read file names/paths in a directory for the agent to process individually.
    
    Args:
        directory_path: Path to the directory to read
        
    Returns:
        Dictionary containing directory info and list of file paths
    """
    try:
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory_path}"}
        
        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}
        
        file_paths = [str(f) for f in sorted(dir_path.iterdir()) if f.is_file()]
        
        return {
            "directory": str(dir_path),
            "file_count": len(file_paths),
            "files": file_paths
        }
    
    except Exception as e:
        return {"error": f"Error reading directory: {str(e)}"}

tools = [read_file_content, write_file_content, read_directory]

# 4. Define Prompt and Agent Executor
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Senior Software Engineer specializing in code refactoring and bug fixes.
    
    Your task is to:
    1. Review the refactoring plan provided by the auditor carefully
    2. For EACH file mentioned in the plan:
       a. Use read_directory to see available files
       b. Use read_file_content to read the current code
       c. Apply ALL the fixes mentioned in the refactoring plan
       d. Use write_file_content to save the corrected code
    3. After fixing all files, provide a summary of what was fixed
    
    IMPORTANT: You MUST actually read and write the files. Do not just describe what you would do.
    Complete ALL fixes before finishing."""),
    ("user", """Refactoring Plan from Auditor:
{output}

Target Directory: {input}

Please implement ALL the fixes described in the refactoring plan above. 
Read each file, apply the corrections, and write the fixed code back."""),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

fixer_agent = create_openai_tools_agent(llm, tools, prompt)
fixer_agent_executor = AgentExecutor(
    agent=fixer_agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
    early_stopping_method="generate"
)

# 5. State and Run Function
from typing_extensions import TypedDict

class AgentState(TypedDict):
    input: str
    output: str

# Import fallback utilities
from src.utils.llm_fallback import FREE_MODELS, NonStreamingChatOpenAI

def run_fixer_agent(state: AgentState):
    """Run fixer agent with automatic model fallback."""
    last_exception = None
    
    for model in FREE_MODELS:
        try:
            fallback_llm = NonStreamingChatOpenAI(
                model=model,
                api_key=OPENROUTER_API_KEY,
                temperature=0,
                base_url="https://openrouter.ai/api/v1",
                streaming=False,
                model_kwargs={},
            )
            
            fallback_agent = create_openai_tools_agent(fallback_llm, tools, prompt)
            fallback_executor = AgentExecutor(
                agent=fallback_agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15,
                early_stopping_method="generate",
            )
            
            response = fallback_executor.invoke(
                {"input": state["input"], "output": state["output"]},
                config={"callbacks": []}
            )
            return {"output": response["output"], "model_used": model}
            
        except Exception as e:
            error_str = str(e).lower()
            last_exception = e
            
            retryable = any(x in error_str for x in [
                "429", "rate limit", "streaming", "tools are not supported", 
                "400", "404", "no endpoints", "tool use", "provider"
            ])
            
            if retryable:
                continue
            else:
                raise e
    
    raise last_exception