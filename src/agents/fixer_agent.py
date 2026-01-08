import os
import subprocess
from pathlib import Path
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
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# FORCE disable streaming in multiple ways
llm.streaming = False
if hasattr(llm, 'model_kwargs'):
    llm.model_kwargs = llm.model_kwargs or {}
    llm.model_kwargs['stream'] = False

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
def list_sandbox_files() -> str:
    """List all Python files in the sandbox directory (recursively).
    Returns a formatted string of file paths relative to sandbox root."""
    try:
        files = [
            str(f.relative_to(SANDBOX_ROOT))
            for f in SANDBOX_ROOT.rglob("*.py")
            if f.is_file()
        ]
        if not files:
            return "No Python files found in sandbox."
        return "Available files:\n" + "\n".join(f"  - {f}" for f in files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

tools = [read_file_content, write_file_content, list_sandbox_files]

# 4. Define Prompt and Agent Executor
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Senior Software Engineer specializing in code refactoring and bug fixes.
    
    Your task is to:
    1. Review the refactoring plan provided by the auditor carefully
    2. For EACH file mentioned in the plan:
       a. Use list_sandbox_files to see available files
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

def run_fixer_agent(state: AgentState):
    response = fixer_agent_executor.invoke(
        {
            "input": state["input"],
            "output": state["output"],
        },
        config={"callbacks": []}
    )
    
    return {
        "output": response["output"]
    }