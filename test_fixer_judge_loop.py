"""
Test file for Fixer-Judge loop workflow.
Entry point: Fixer Agent
Loop: Fixer -> Judge -> (Pass? END : Fixer)
"""

import os
import argparse
from dotenv import load_dotenv
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import agents
from src.agents.fixer_agent import run_fixer_agent
from src.agents.judge_agent import run_judge_agent

load_dotenv()

# State definition
class AgentState(TypedDict):
    input: str           # Target directory
    output: str          # Current output/report
    iteration: int       # Track iterations
    test_passed: bool    # Whether tests passed


def fixer_node(state: AgentState):
    """Fixer agent node - fixes code based on refactoring plan."""
    print("\n🔧 Running Fixer Agent...")
    result = run_fixer_agent({
        "input": state["input"],
        "output": state["output"],
    })
    return {
        "output": result["output"],
        "iteration": state.get("iteration", 0) + 1,
    }


def judge_node(state: AgentState):
    """Judge agent node - creates and runs tests."""
    print("\n⚖️ Running Judge Agent...")
    result = run_judge_agent({
        "input": state["input"],
        "output": state["output"],
    })
    
    # Check if tests passed by looking for common pytest success indicators
    output = result["output"].lower()
    test_passed = (
        "passed" in output and "failed" not in output
    ) or (
        "all tests passed" in output
    ) or (
        "100%" in output and "failed" not in output
    )
    
    return {
        "output": result["output"],
        "test_passed": test_passed,
    }


def should_continue(state: AgentState) -> str:
    """
    Conditional edge: decide whether to continue loop or end.
    Returns 'end' if tests passed or max iterations reached.
    Returns 'fixer' if tests failed and need more fixes.
    """
    max_iterations = 3
    
    if state.get("test_passed", False):
        print("\n✅ Tests PASSED! Ending workflow.")
        return "end"
    
    if state.get("iteration", 0) >= max_iterations:
        print(f"\n⚠️ Max iterations ({max_iterations}) reached. Ending workflow.")
        return "end"
    
    print(f"\n❌ Tests FAILED. Returning to Fixer (iteration {state.get('iteration', 0)})...")
    # Prepare feedback for fixer
    return "fixer"


def prepare_feedback(state: AgentState):
    """Prepare feedback from judge for the fixer agent."""
    import time
    print("\n⏳ Waiting 60 seconds before next iteration (rate limit protection)...")
    time.sleep(60)
    
    feedback = f"""
=== TEST FAILURE FEEDBACK ===
The following issues were found during testing:

{state['output']}

Please fix the code to make the tests pass. Focus on:
1. Any runtime errors mentioned
2. Assertion failures
3. Missing functionality
"""
    return {"output": feedback}


def build_graph():
    """Build the Fixer-Judge loop graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fixer", fixer_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("prepare_feedback", prepare_feedback)
    
    # Set entry point
    workflow.set_entry_point("fixer")
    
    # Add edges
    workflow.add_edge("fixer", "judge")
    
    # Conditional edge from judge
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "end": END,
            "fixer": "prepare_feedback",
        }
    )
    
    # Feedback goes back to fixer
    workflow.add_edge("prepare_feedback", "fixer")
    
    return workflow.compile(checkpointer=MemorySaver())


# Simple buggy file content for testing
BUGGY_FILE_CONTENT = '''"""Simple buggy module for testing the fixer-judge loop."""


def add(a, b):
    """Add two numbers."""
    # Bug 1: returns subtraction instead of addition
    return a - b


def multiply(a, b):
    """Multiply two numbers."""
    # Bug 2: returns addition instead of multiplication
    return a + b
'''

# Simple refactoring plan (only fixing ONE bug)
SIMPLE_REFACTORING_PLAN = '''
=== PARTIAL REFACTORING PLAN ===

Target: test_loop/buggy.py

Fix ONLY this one bug:

1. add function (line 7):
   - BUG: returns a - b instead of a + b
   - FIX: change "return a - b" to "return a + b"

DO NOT fix any other issues. The multiply function bug will be found by tests later.
'''


def setup_test_file(target_dir: str):
    """Create the buggy test file."""
    import os
    from pathlib import Path
    
    dir_path = Path(target_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    file_path = dir_path / "buggy.py"
    file_path.write_text(BUGGY_FILE_CONTENT)
    print(f"📝 Created buggy file at: {file_path}")
    return str(file_path)


def main():
    parser = argparse.ArgumentParser(description="Test Fixer-Judge Loop")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./sandbox/test_loop",
        help="Target directory for the test"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup the buggy test file before running"
    )
    args = parser.parse_args()
    
    # Setup test file if requested
    if args.setup:
        setup_test_file(args.target_dir)
    
    print("🚀 Starting Fixer-Judge Loop Test")
    print(f"📁 Target directory: {args.target_dir}")
    print("=" * 50)
    
    # Build and run the graph
    app = build_graph()
    
    config = {"configurable": {"thread_id": "fixer-judge-loop-test"}}
    
    # Initial state with the partial refactoring plan
    initial_state = {
        "input": args.target_dir,
        "output": SIMPLE_REFACTORING_PLAN,
        "iteration": 0,
        "test_passed": False,
    }
    
    try:
        result = app.invoke(initial_state, config=config)
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULT")
        print("=" * 50)
        print(f"Iterations: {result.get('iteration', 'N/A')}")
        print(f"Tests Passed: {result.get('test_passed', False)}")
        print("\nFinal Output:")
        print(result.get("output", "No output"))
        
    except Exception as e:
        print(f"\n❌ Error during workflow: {e}")
        raise


if __name__ == "__main__":
    main()
