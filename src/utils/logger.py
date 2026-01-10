import json
import os
import uuid
from datetime import datetime
from enum import Enum

# Chemin du fichier de logs
LOG_FILE = os.path.join("logs", "experiment_data.json")

class ActionType(str, Enum):
    """
    Énumération des types d'actions possibles pour standardiser l'analyse.
    """
    STARTUP = "STARTUP"         # Experiment initialization
    ANALYSIS = "CODE_ANALYSIS"  # Audit, lecture, recherche de bugs
    GENERATION = "CODE_GEN"     # Création de nouveau code/tests/docs
    DEBUG = "DEBUG"             # Analyse d'erreurs d'exécution
    FIX = "FIX"                 # Application de correctifs
    TEST = "TEST"               # Running tests (Judge)
    COMPLETION = "COMPLETION"   # Experiment completed


def log_startup(target_dir: str, models: list):
    """
    Log the start of an experiment session.
    
    Args:
        target_dir: The target directory being analyzed
        models: List of models available for fallback
    """
    os.makedirs("logs", exist_ok=True)
    
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "agent": "SYSTEM",
        "model": None,
        "action": ActionType.STARTUP.value,
        "details": {
            "target_directory": target_dir,
            "available_models": models,
            "session_start": datetime.now().isoformat()
        },
        "status": "STARTED"
    }
    
    _write_log_entry(entry)
    return entry["id"]  # Return session ID for tracking


def log_agent_interaction(agent_name: str, model_used: str, action: ActionType, 
                          input_prompt: str, output_response: str, status: str,
                          iteration: int = None, extra_details: dict = None):
    """
    Log an LLM-agent interaction with full context.
    
    Args:
        agent_name: Name of the agent (Auditor, Fixer, Judge)
        model_used: The LLM model that was used
        action: ActionType enum value
        input_prompt: The input/prompt sent to the agent
        output_response: The response from the agent
        status: SUCCESS or FAILURE
        iteration: Current iteration number (for loop tracking)
        extra_details: Any additional context to log
    """
    details = {
        "input_prompt": input_prompt[:2000] if input_prompt else "",  # Truncate for storage
        "output_response": output_response[:5000] if output_response else "",
        "response_length": len(output_response) if output_response else 0,
    }
    
    if iteration is not None:
        details["iteration"] = iteration
    
    if extra_details:
        details.update(extra_details)
    
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "model": model_used,
        "action": action.value if isinstance(action, ActionType) else action,
        "details": details,
        "status": status
    }
    
    _write_log_entry(entry)


def log_completion(iterations: int, test_passed: bool, final_output: str):
    """
    Log the completion of an experiment session.
    
    Args:
        iterations: Total number of iterations completed
        test_passed: Whether tests passed at the end
        final_output: The final output summary
    """
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "agent": "SYSTEM",
        "model": None,
        "action": ActionType.COMPLETION.value,
        "details": {
            "total_iterations": iterations,
            "test_passed": test_passed,
            "final_output": final_output[:3000] if final_output else "",
            "session_end": datetime.now().isoformat()
        },
        "status": "SUCCESS" if test_passed else "COMPLETED_WITH_FAILURES"
    }
    
    _write_log_entry(entry)


def _write_log_entry(entry: dict):
    """Internal function to write a log entry to file."""
    os.makedirs("logs", exist_ok=True)
    
    data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
        except json.JSONDecodeError:
            data = []
    
    data.append(entry)
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)