import os
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
# This will be used to restrict the use of the tools to the sandbox directory (security measure)
SANDBOX_ROOT = Path(".").resolve()

# 2. Initialize LLM (Streaming must be False for OpenRouter tool calls)
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    openai_api_key=OPENROUTER_API_KEY,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    streaming=False
)
llm.streaming = False  # Ensure streaming is disabled

# 3. Define Tools
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
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file_content(file_path: str, content: str) -> str:
    """Writes content to a file on the local disk. 
    Use this to save corrected or refactored code."""
    try:
        safe_path = (SANDBOX_ROOT / file_path).resolve()

        # Security check: ensure the test file is within the sandbox
        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote corrected code to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

@tool
def list_sandbox_files() -> list[str]:
    """List all Python files in the sandbox directory (recursively)."""
    try:
        files = [
            str(f.relative_to(SANDBOX_ROOT))
            for f in SANDBOX_ROOT.rglob("*.py")
            if f.is_file()
        ]
        return files
    except Exception as e:
        return [f"Error listing files: {str(e)}"]
    
tools = [read_file_content, write_file_content, list_sandbox_files]
# 4. Define Prompt and Agent Executor
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Senior Software Engineer. 
    Your task is to:
    1. Read all buggy Python files in the input directory.
    2. Read the corresponding refactoring plan files located in './sandbox/<original_filename>_refactor_plan.txt'.
    3. Apply the fixes described in the refactoring plans **directly to the original source files** (overwrite the original files).
    4. Generate pytest functions for each corrected file and write them to './sandbox/test_runner.py'. 
    - The tests should **call the fixed functions**, do not redefine them.
    5. For each corrected file, return the filename and full corrected code.
    Always ensure your output is structured and complete for each file, and do not create any separate corrected copies. The original source files should be fully updated."""),  
        MessagesPlaceholder(variable_name="agent_scratchpad"),
])

fixer_agent = create_openai_tools_agent(llm, tools, prompt)
fixer_agent_executor = AgentExecutor(agent=fixer_agent, tools=tools, verbose=True)

# 5. DEFINE LANGGRAPH 0.0.25 LOGIC
class AgentState(TypedDict):
    input: str
    output: str

def run_fixer_agent(state: AgentState):
    response = fixer_agent_executor.invoke(
        {
            "input": state["input"]
        },
        # This config helps override legacy streaming behaviors
        config={"callbacks": []} 
    )

    return {
        "output": response["output"]
    }
