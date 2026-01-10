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

from src.prompts.auditor_prompts import get_auditor_system_prompt, get_auditor_user_prompt
from src.utils.logger import log_experiment, ActionType


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

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote corrected code to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


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

tools = [read_file_content, write_file_content, run_pylint]

# 4. Define Prompt and Agent Executor

def create_auditor_agent(target_dir: str):
    #create agent with a dynamic prompt
    system_prompt = get_auditor_system_prompt(target_dir)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_auditor_agent(target_dir: str) -> dict:
    agent_executor = create_auditor_agent(target_dir)
    user_input = get_auditor_user_prompt(target_dir)
    
    # Log before execution
    log_experiment(
        agent_name="Auditor_Agent",
        model_used="meta-llama/llama-3.3-70b-instruct:free",
        action=ActionType.ANALYSIS,
        details={
            "input_prompt": user_input,
            "target_directory": target_dir,
            "output_response": ""  # later
        },
        status="IN_PROGRESS"
    )
    
    try:
        result = agent_executor.invoke({"input": user_input})
        
        # Log success
        log_experiment(
            agent_name="Auditor_Agent",
            model_used="meta-llama/llama-3.3-70b-instruct:free",
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": user_input,
                "output_response": result["output"],
                "target_directory": target_dir
            },
            status="SUCCESS"
        )
        
        return {"output": result["output"], "status": "SUCCESS"}
    
    except Exception as e:
        # Log failure
        log_experiment(
            agent_name="Auditor_Agent",
            model_used="meta-llama/llama-3.3-70b-instruct:free",
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": user_input,
                "output_response": str(e)
            },
            status="FAILURE"
        )
        
        return {"output": str(e), "status": "FAILURE"}

# 5. DEFINE LANGGRAPH 0.0.25 LOGIC
class AgentState(TypedDict):
    input: str
    output: str

def run_auditor_agent_langgraph(state: AgentState):

    result = run_auditor_agent(state["input"])
    
    return {"output": result["output"]}
