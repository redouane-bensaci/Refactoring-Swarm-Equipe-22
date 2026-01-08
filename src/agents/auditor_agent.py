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
    raise ValueError("OPENROUTER_API_KEY2 environment variable is not set")
# This will be used to restrict the use of the tools to the sandbox directory (security measure)
SANDBOX_ROOT = Path(".").resolve()

# 2. Initialize LLM (Streaming must be False for OpenRouter tool calls)
llm = ChatOpenAI(
    model= "meta-llama/llama-3.3-70b-instruct:free",
    streaming=False,
    api_key=OPENROUTER_API_KEY,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)
llm.streaming = False  # Ensure streaming is disabled

# 3. Define Tools

@tool
# ...existing code...

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

@tool
def read_file_content(file_path: str) -> str:
    """Reads the content of a file from the local disk."""
    try:
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
    """Runs pylint on the given file and returns the output as a string."""
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    try:
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
        
        # Run pylint as a subprocess
        result = subprocess.run(
            ["pylint", "--output-format=text", file_path],
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
    ("user", "Analyze the directory: {input}"),  # The input contains the directory path
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

def run_auditor_agent(state: AgentState):

    response = auditor_agent_executor.invoke(
        {
            "input": state["input"],
        },
        # This config helps override legacy streaming behaviors
        config={"callbacks": []} 
    )

    return {
        "output": response["output"]
    }
