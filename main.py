import os
import sys
import argparse
from dotenv import load_dotenv
from typing import TypedDict

from src.utils.logger import log_startup, log_agent_interaction, log_completion, ActionType
from src.agents.fixer_agent import run_fixer_agent  
from src.agents.auditor_agent import run_auditor_agent
from src.agents.judge_agent import run_judge_agent
from src.utils.llm_fallback import FREE_MODELS

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
    model_used: str      # Track which model was used

# -----------------------------
# Wrap agents for graph nodes
# -----------------------------
def auditor_node(state: AgentState):
    input_prompt = f"Analyze directory: {state['input']}"
    result = run_auditor_agent({"input": state["input"], "output": ""})
    
    log_agent_interaction(
        agent_name="Auditor",
        model_used=result.get("model_used", FREE_MODELS[0]),
        action=ActionType.ANALYSIS,
        input_prompt=input_prompt,
        output_response=result["output"],
        status="SUCCESS",
        iteration=0
    )
    
    return {
        "input": state["input"],
        "output": result["output"],
        "iteration": 0,
        "test_passed": False,
        "model_used": result.get("model_used", ""),
    }

def fixer_node(state: AgentState):
    input_prompt = f"Fix code based on: {state['output'][:500]}"
    result = run_fixer_agent({
        "input": state["input"],
        "output": state["output"]
    })
    
    iteration = state.get("iteration", 0) + 1
    log_agent_interaction(
        agent_name="Fixer",
        model_used=result.get("model_used", FREE_MODELS[0]),
        action=ActionType.FIX,
        input_prompt=input_prompt,
        output_response=result["output"],
        status="SUCCESS",
        iteration=iteration
    )
    
    return {
        "input": state["input"],
        "output": result["output"],
        "iteration": iteration,
        "test_passed": state.get("test_passed", False),
        "model_used": result.get("model_used", ""),
    }

def judge_node(state: AgentState):
    input_prompt = f"Verify fixes in: {state['input']}"
    result = run_judge_agent({
        "input": state["input"],
        "output": state["output"]
    })
    
    # Check if tests passed using the VERDICT signal
    output = result["output"]
    output_lower = output.lower()
    
    # Primary: Look for explicit VERDICT
    if "verdict: all_tests_passed" in output_lower:
        test_passed = True
    elif "verdict: tests_failed" in output_lower:
        test_passed = False
    else:
        # Fallback: Parse pytest output format "X passed" with no failures
        import re
        passed_match = re.search(r'(\d+)\s+passed', output_lower)
        failed_match = re.search(r'(\d+)\s+failed', output_lower)
        
        if passed_match:
            passed_count = int(passed_match.group(1))
            failed_count = int(failed_match.group(1)) if failed_match else 0
            test_passed = passed_count > 0 and failed_count == 0
        else:
            test_passed = False
    
    log_agent_interaction(
        agent_name="Judge",
        model_used=result.get("model_used", FREE_MODELS[0]),
        action=ActionType.TEST,
        input_prompt=input_prompt,
        output_response=result["output"],
        status="SUCCESS" if test_passed else "TESTS_FAILED",
        iteration=state.get("iteration", 0),
        extra_details={"test_passed": test_passed}
    )
    
    return {
        "input": state["input"],
        "output": result["output"],
        "iteration": state.get("iteration", 0),
        "test_passed": test_passed,
        "model_used": result.get("model_used", ""),
    }

def should_continue(state: AgentState) -> str:
    """
    Conditional edge: decide whether to continue loop or end.
    Returns 'end' if tests passed or max iterations reached.
    Returns 'prepare_feedback' if tests failed and need more fixes.
    """
    max_iterations = 3
    
    if state.get("test_passed", False):
        return "end"
    
    if state.get("iteration", 0) >= max_iterations:
        return "end"
    
    return "prepare_feedback"

def prepare_feedback(state: AgentState):
    """Prepare feedback from judge for the fixer agent."""
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
        sys.exit(1)

    # Log experiment startup
    log_startup(target_dir=args.target_dir, models=FREE_MODELS)

    # -----------------------------
    # Run the workflow
    # -----------------------------
    config = {"configurable": {"thread_id": "session_1"}}
    
    result = app.invoke({
        "input": args.target_dir,
        "output": "",
        "iteration": 0,
        "test_passed": False,
        "model_used": "",
    }, config=config)
    
    # Log experiment completion
    log_completion(
        iterations=result.get('iteration', 0),
        test_passed=result.get('test_passed', False),
        final_output=result['output']
    )
    
    print(f"\nIterations: {result.get('iteration', 'N/A')}")
    print(f"Tests Passed: {result.get('test_passed', False)}")
    print(f"\nFinal Output:\n{result['output']}\n")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()