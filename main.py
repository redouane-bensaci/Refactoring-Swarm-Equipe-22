import os
import sys
import argparse
import time
from dotenv import load_dotenv
from typing import TypedDict

from src.utils.logger import log_experiment
from src.services.file_handler import file_service
from src.services.static_analyzer import static_analyzer_service
from src.agents.fixer_agent import run_fixer_agent  
from src.agents.auditor_agent import run_auditor_agent
from src.agents.judge_agent import run_judge_agent

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Define Graph State
# -----------------------------
class AgentState(TypedDict):
    input: str           # Target directory
    output: str          # Current output/report
    iteration: int       # Track iterations for fixer-judge loop
    test_passed: bool    # Whether tests passed

# -----------------------------
# Wrap agents for graph nodes
# -----------------------------
def auditor_node(state: AgentState):
    print("\n🔍 Running Auditor Agent...")
    result = run_auditor_agent({"input": state["input"], "output": ""})
    return {
        "input": state["input"],
        "output": result["output"],
        "iteration": 0,
        "test_passed": False,
    }

def fixer_node(state: AgentState):
    print("\n🔧 Running Fixer Agent...")
    result = run_fixer_agent({
        "input": state["input"],
        "output": state["output"]
    })
    return {
        "input": state["input"],
        "output": result["output"],
        "iteration": state.get("iteration", 0) + 1,
        "test_passed": state.get("test_passed", False),
    }

def judge_node(state: AgentState):
    print("\n⚖️ Running Judge Agent...")
    result = run_judge_agent({
        "input": state["input"],
        "output": state["output"]
    })
    
    # Check if tests passed by looking for common pytest success indicators
    output = result["output"].lower()
    test_passed = (
        ("passed" in output and "failed" not in output) or
        ("all tests passed" in output) or
        ("100%" in output and "failed" not in output)
    )
    
    return {
        "input": state["input"],
        "output": result["output"],
        "iteration": state.get("iteration", 0),
        "test_passed": test_passed,
    }

def should_continue(state: AgentState) -> str:
    """
    Conditional edge: decide whether to continue loop or end.
    Returns 'end' if tests passed or max iterations reached.
    Returns 'prepare_feedback' if tests failed and need more fixes.
    """
    max_iterations = 3
    
    if state.get("test_passed", False):
        print("\n✅ Tests PASSED! Ending workflow.")
        return "end"
    
    if state.get("iteration", 0) >= max_iterations:
        print(f"\n⚠️ Max iterations ({max_iterations}) reached. Ending workflow.")
        return "end"
    
    print(f"\n❌ Tests FAILED. Returning to Fixer (iteration {state.get('iteration', 0)})...")
    return "prepare_feedback"

def prepare_feedback(state: AgentState):
    """Prepare feedback from judge for the fixer agent with rate limit delay."""
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
    return {
        "input": state["input"],
        "output": feedback,
        "iteration": state.get("iteration", 0),
        "test_passed": False,
    }

# -----------------------------
# Build LangGraph Workflow with Fixer-Judge Loop
# -----------------------------
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("auditor_agent", auditor_node)
workflow.add_node("fixer_agent", fixer_node)
workflow.add_node("judge_agent", judge_node)
workflow.add_node("prepare_feedback", prepare_feedback)

# Set entry point
workflow.set_entry_point("auditor_agent")

# Auditor -> Fixer (first pass)
workflow.add_edge("auditor_agent", "fixer_agent")

# Fixer -> Judge
workflow.add_edge("fixer_agent", "judge_agent")

# Judge -> Conditional (Loop or End)
workflow.add_conditional_edges(
    "judge_agent",
    should_continue,
    {
        "end": END,
        "prepare_feedback": "prepare_feedback",
    }
)

# Feedback -> Fixer (loop back)
workflow.add_edge("prepare_feedback", "fixer_agent")

# -----------------------------
# Persistence
# -----------------------------
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the LangGraph pipeline: Auditor -> Fixer <-> Judge Loop")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to analyze")
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        print(f"❌ Target directory '{args.target_dir}' not found.")
        sys.exit(1)

    print(f"🚀 Starting workflow on: {args.target_dir}")
    print("📋 Pipeline: Auditor -> Fixer <-> Judge (loop until tests pass)")

    # -----------------------------
    # Run the workflow
    # -----------------------------
    print("\n✅ LangGraph workflow running...")
    config = {"configurable": {"thread_id": "session_1"}}
    
    # Initialize with all state fields
    result = app.invoke({
        "input": args.target_dir,
        "output": "",
        "iteration": 0,
        "test_passed": False,
    }, config=config)
    
    print(f"\n{'='*50}")
    print("📊 FINAL RESULT")
    print(f"{'='*50}")
    print(f"Iterations: {result.get('iteration', 'N/A')}")
    print(f"Tests Passed: {result.get('test_passed', False)}")
    print(f"\n💡 Final Output:\n{result['output']}\n")

    # -----------------------------
    # Optional: Static analysis log
    # -----------------------------
    print("🔍 Static Analysis Summary:")
    for root, _, files in os.walk(args.target_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                analysis = static_analyzer_service.analyze(path)
                print(f"- {file}: {len(analysis.issues)} issues detected")
                for issue in analysis.issues:
                    print(f"  • {issue}")

    print("\n✅ Mission Complete!")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()