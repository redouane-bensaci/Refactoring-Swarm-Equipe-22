import os
import subprocess
from dotenv import load_dotenv
from typing_extensions import TypedDict
from pathlib import Path

# Core LangChain 0.1.10 imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Setup Environment
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY3")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY3 environment variable is not set")

SANDBOX_ROOT = (Path(__file__).parent.parent.parent / "sandbox").resolve()

# 2. Initialize LLM - CRITICAL: No streaming
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

# FORCE disable streaming
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
            return f"File not found: {file_path}"

        if not safe_path.is_file():
            return f"Not a file: {file_path}"
        
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Writes content to a file (for creating test files).
    
    Args:
        file_path: Relative path from sandbox root (e.g., 'test3/test_add.py')
        content: The test code to write
    """
    try:
        if file_path.startswith('sandbox/'):
            file_path = file_path.replace('sandbox/', '', 1)
            
        safe_path = (SANDBOX_ROOT / file_path).resolve()

        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return f"✅ Successfully wrote test file to {file_path}"
    except Exception as e:
        return f"❌ Error writing file: {str(e)}"

@tool
def run_pytest(test_path: str) -> str:
    """Runs pytest on the specified test file or directory.
    
    Args:
        test_path: Path to test file or directory (e.g., 'test3/test_add.py' or 'test3/')
    """
    try:
        if test_path.startswith('sandbox/'):
            test_path = test_path.replace('sandbox/', '', 1)
            
        safe_path = (SANDBOX_ROOT / test_path).resolve()

        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"
        
        if not safe_path.exists():
            return f"Test path not found: {test_path}"
        
        # Run pytest with verbose output and coverage
        result = subprocess.run(
            ["pytest", str(safe_path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=SANDBOX_ROOT
        )
        
        output = f"PYTEST RESULTS:\n{result.stdout}\n"
        if result.stderr:
            output += f"\nERRORS:\n{result.stderr}"
        
        return output
    except Exception as e:
        return f"Error running pytest: {str(e)}"

@tool
def list_directory_files(directory_path: str) -> str:
    """List all files in a directory.
    
    Args:
        directory_path: Relative path from sandbox root
    """
    try:
        if directory_path.startswith('sandbox/'):
            directory_path = directory_path.replace('sandbox/', '', 1)
            
        safe_path = (SANDBOX_ROOT / directory_path).resolve()

        if not safe_path.is_relative_to(SANDBOX_ROOT):
            return "Access denied: path outside sandbox"
        
        if not safe_path.exists():
            return f"Directory not found: {directory_path}"
        
        if not safe_path.is_dir():
            return f"Not a directory: {directory_path}"
        
        files = [f.name for f in safe_path.iterdir() if f.is_file()]
        return "Files in directory:\n" + "\n".join(f"  - {f}" for f in files)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

tools = [read_file_content, write_file, run_pytest, list_directory_files]

# 4. Define Prompt and Agent Executor
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Judge Agent specializing in Test-Driven Development (TDD) and code quality verification.

Your task:
1. **Read the corrected code files** from the target directory using read_file_content tool
2. **Analyze each corrected file** and create comprehensive unit tests following TDD principles:
   - Test edge cases and boundary conditions
   - Test normal operation scenarios
   - Test error handling and exceptions
   - Ensure good test coverage
3. **Write the unit tests** to appropriate test files (e.g., 'test_<filename>.py') using write_file tool
   - Use pytest framework
   - Follow pytest naming conventions (test_*.py or *_test.py)
   - Include proper assertions and test fixtures
4. **Run the unit tests** using the run_pytest tool to verify the corrected code
5. **Generate a comprehensive report** including:
   - Test results (passed/failed/skipped)
   - Code coverage metrics if available
   - Any bugs or issues discovered during testing
   - Quality assessment of the corrected code
   - Recommendations for further improvements

Remember: You are verifying that the Fixer Agent's corrections work correctly through automated testing.
Write thorough tests that would catch potential bugs."""),
    ("user", """Fixer Agent Report:
{output}

Target Directory: {input}

Please create and run unit tests for all corrected files in the directory above."""),  # FIXED: Added {output} placeholder
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

judge_agent = create_openai_tools_agent(llm, tools, prompt)
judge_agent_executor = AgentExecutor(
    agent=judge_agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,  # Tests need more iterations
    early_stopping_method="generate"
)

# 5. State and Run Function
class AgentState(TypedDict):
    input: str
    output: str

def run_judge_agent(state: AgentState):
    response = judge_agent_executor.invoke(
        {
            "input": state["input"],
            "output": state["output"],  # FIXED: Pass the fixer's output
        },
        config={"callbacks": []} 
    )

    return {
        "output": response["output"]
    }