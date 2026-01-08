"""Swarm Orchestrator - Multi-agent coordination using LangGraph.

Coordinates Auditor, Fixer, and Judge agents for code refactoring.
"""

# Import agent runners
from src.agents.auditor_agent import run_auditor_agent
from src.agents.fixer_agent import run_fixer_agent
from src.agents.judge_agent import run_judge_agent

# Simple sequential execution for testing
def run_swarm_sequential(initial_state):
    print("DEBUG: Running sequential swarm", flush=True)

    # Run auditor
    print("DEBUG: Running auditor", flush=True)
    state = run_auditor_agent(initial_state)
    print("DEBUG: Auditor completed", flush=True)

    # Run fixer
    print("DEBUG: Running fixer", flush=True)
    state = run_fixer_agent(state)
    print("DEBUG: Fixer completed", flush=True)

    # Run judge
    print("DEBUG: Running judge", flush=True)
    state = run_judge_agent(state)
    print("DEBUG: Judge completed", flush=True)

    return state

app = run_swarm_sequential
