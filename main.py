import os
import sys
import argparse
from dotenv import load_dotenv
from typing import TypedDict

from src.utils.logger import log_experiment
from src.services.file_handler import file_service
from src.services.static_analyzer import static_analyzer_service
from src.agents.fixer_agent import run_fixer_agent  
from src.agents.auditor_agent import run_auditor_agent
from src.agents.judge_agent import run_judge_agent

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Define Graph State
# -----------------------------
class AgentState(TypedDict):
    input: str  
    output: str

# -----------------------------
# Wrap agents for graph nodes
# -----------------------------
def auditor_node(state: AgentState):
    print("🔍 Running Auditor Agent...")
    result = run_auditor_agent({"input": state["input"], "output": ""})
    return {
        "input": state["input"],           # Keep the directory path
        "output": result["output"]         # FIX: Extract output from result dict
    }

def fixer_node(state: AgentState):
    print("🔧 Running Fixer Agent...")
    result = run_fixer_agent({
        "input": state["input"],           # Directory path
        "output": state["output"]          # FIX: Pass auditor's refactoring plan
    })
    return {
        "input": state["input"],           # Keep the directory path
        "output": result["output"]         # FIX: Extract output from result dict
    }

def judge_node(state: AgentState):
    print("⚖️ Running Judge Agent...")
    result = run_judge_agent({
        "input": state["input"],           # Directory path
        "output": state["output"]          # FIX: Pass fixer's report
    })
    return {
        "input": state["input"],           # Keep the directory path
        "output": result["output"]         # FIX: Extract output from result dict
    }

# -----------------------------
# Build LangGraph Workflow
# -----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("auditor_agent", auditor_node)
workflow.add_node("fixer_agent", fixer_node)
workflow.add_node("judge_agent", judge_node)

# Linear flow: Auditor -> Fixer -> Judge -> End
workflow.set_entry_point("auditor_agent")
workflow.add_edge("auditor_agent", "fixer_agent")
workflow.add_edge("fixer_agent", "judge_agent")
workflow.add_edge("judge_agent", "__end__")

# -----------------------------
# Persistence
# -----------------------------
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the LangGraph pipeline: Auditor -> Fixer -> Judge")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to analyze")
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        print(f"❌ Target directory '{args.target_dir}' not found.")
        sys.exit(1)

    print(f"🚀 Starting workflow on: {args.target_dir}")

    # -----------------------------
    # Run the workflow
    # -----------------------------
    print("✅ LangGraph workflow running...")
    config = {"configurable": {"thread_id": "session_1"}}
    
    # FIX: Initialize with both input AND output fields
    result = app.invoke({"input": args.target_dir, "output": ""}, config=config)
    
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

    print("✅ Mission Complete!")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()