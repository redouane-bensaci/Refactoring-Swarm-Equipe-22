import os
import subprocess
from pathlib import Path
from typing import Union, Dict, List
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from pathlib import Path

# Core LangChain 0.1.10 imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangGraph 0.0.25 specific imports
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 1. Setup Environment
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

# This will be used to restrict the use of the tools to the sandbox directory (security measure)
SANDBOX_ROOT = (Path(__file__).parent.parent.parent / "sandbox").resolve()

# 2. Initialize LLM (Streaming must be False for OpenRouter tool calls)
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
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    streaming=False,
    model_kwargs={},
)

# 3. Define Tools

@tool
def read_directory(directory_path: str) -> Dict[str, Union[str, int, List[str]]]:
    """
    Read file names/paths in a directory for the agent to process individually.
    
    Args:
        directory_path: Path to the directory to read (e.g., './sandbox/test3' or 'test3')
        
    Returns:
        Dictionary containing directory info and list of file paths
    """
    try:
        # Handle different path formats
        if directory_path.startswith('./sandbox/'):
            directory_path = directory_path.replace('./sandbox/', '', 1)
        elif directory_path.startswith('sandbox/'):
            directory_path = directory_path.replace('sandbox/', '', 1)
        
        # Use SANDBOX_ROOT as base
        dir_path = (SANDBOX_ROOT / directory_path).resolve()
        
        # Security check
        if not dir_path.is_relative_to(SANDBOX_ROOT):
            return {"error": f"Access denied: path outside sandbox"}
        
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory_path}"}
        
        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}
        
        file_paths = [str(f.relative_to(SANDBOX_ROOT)) for f in sorted(dir_path.iterdir()) if f.is_file()]
        
        return {
            "directory": str(dir_path.relative_to(SANDBOX_ROOT)),
            "file_count": len(file_paths),
            "files": file_paths
        }
    
    except Exception as e:
        return {"error": f"Error reading directory: {str(e)}"}

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

        # Security check: ensure the test file is within the sandbox
        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"
        
        # Check if the test file exists
        if not safe_path.exists():
            return f"File not found: {file_path}"

        # Check if it's a file
        if not safe_path.is_file():
            return f"Not a file: {file_path}"
        
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def run_pylint(file_path: str) -> str:
    """Runs pylint on the given file and returns the output as a string.
    
    Args:
        file_path: Relative path from sandbox root (e.g., 'test3/add.py')
    """
    try:
        if file_path.startswith('sandbox/'):
            file_path = file_path.replace('sandbox/', '', 1)
            
        safe_path = (SANDBOX_ROOT / file_path).resolve()

        # Security check: ensure the test file is within the sandbox
        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"
        
        # Check if the test file exists
        if not safe_path.exists():
            return f"File not found: {file_path}"

        # Check if it's a file
        if not safe_path.is_file():
            return f"Not a file: {file_path}"
        
        # Run pylint as a subprocess using the resolved path
        result = subprocess.run(
            ["pylint", "--output-format=text", str(safe_path)],
            capture_output=True,
            text=True
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running pylint: {str(e)}"

tools = [read_directory, read_file_content, run_pylint]

# 4. Define Prompt and Agent Executor
# 4. Define Prompt and Agent Executor
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Senior Software Engineer. 
    Your task is to:
    1. Read all files in the target directory using the tool read_directory and read_file_content.
    2. For each file, run static analysis with pylint using the tool run_pylint.
    3. Generate a refactoring plan for the files.
    4. Include pylint messages in the plan to highlight issues.
    5. Include the pylint score at the end in the output.
    Always output the full refactoring plan when writing."""),
    ("user", "Analyze the target directory: {input}"),  # The input contains the directory path
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

auditor_agent = create_openai_tools_agent(llm, tools, prompt)
auditor_agent_executor = AgentExecutor(
    agent=auditor_agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True  # Also add this for better error handling
)
# 5. DEFINE LANGGRAPH 0.0.25 LOGIC
class AgentState(TypedDict):
    input: str
    output: str

# Import fallback utilities
from src.utils.llm_fallback import FREE_MODELS, NonStreamingChatOpenAI

def run_auditor_agent(state: AgentState):
    """Run auditor agent with automatic model fallback."""
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
            )
            
            response = fallback_executor.invoke(
                {"input": state["input"]},
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
