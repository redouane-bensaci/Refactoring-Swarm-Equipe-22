from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Dict, Any
from enum import Enum
import uuid


class ActionType(str, Enum):
    """
    Énumération des types d'actions possibles pour standardiser l'analyse.
    Must match the enum in src/utils/logger.py exactly.
    """
    ANALYSIS = "CODE_ANALYSIS"  # Audit, lecture, recherche de bugs
    GENERATION = "CODE_GEN"     # Création de nouveau code/tests/docs
    DEBUG = "DEBUG"             # Analyse d'erreurs d'exécution
    FIX = "FIX"                 # Application de correctifs


class ExperimentLogEntry(BaseModel):
    """
    Pydantic model for experiment log entries.
    Matches the exact structure required by the documentation and logger.py.
    
    Required fields per documentation:
    - id: Unique identifier (UUID)
    - timestamp: ISO format timestamp
    - agent: Agent name
    - model: LLM model used
    - action: ActionType enum value
    - details: Dictionary containing input_prompt and output_response (mandatory for ANALYSIS, GENERATION, DEBUG, FIX)
    - status: "SUCCESS" or "FAILURE"
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the log entry"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO format timestamp"
    )
    agent: str = Field(
        ...,
        description="Name of the agent (e.g., 'Auditor_Agent', 'Fixer_Agent')"
    )
    model: str = Field(
        ...,
        description="LLM model used (e.g., 'gemini-1.5-flash')"
    )
    action: ActionType = Field(
        ...,
        description="Type of action performed (must use ActionType enum)"
    )
    details: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing action details. MUST include 'input_prompt' and 'output_response' for ANALYSIS, GENERATION, DEBUG, FIX actions"
    )
    status: str = Field(
        ...,
        pattern="^(SUCCESS|FAILURE)$",
        description="Status of the action: SUCCESS or FAILURE"
    )
    
    @model_validator(mode='after')
    def validate_details(self) -> 'ExperimentLogEntry':
        """
        Validate that details contains required fields for specific actions.
        Per documentation: input_prompt and output_response are MANDATORY for:
        - CODE_ANALYSIS (ANALYSIS)
        - CODE_GEN (GENERATION)
        - DEBUG
        - FIX
        """
        action = self.action
        
        if action in [ActionType.ANALYSIS, ActionType.GENERATION, ActionType.DEBUG, ActionType.FIX]:
            required_keys = ["input_prompt", "output_response"]
            missing_keys = [key for key in required_keys if key not in self.details]
            
            if missing_keys:
                raise ValueError(
                    f"❌ Erreur de Logging: Les champs {missing_keys} sont manquants dans 'details'. "
                    f"Ils sont OBLIGATOIRES pour l'action {action.value}."
                )
        
        return self
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "id": "9e82e9b0-9b43-4a78-af43-d5d5ef848a2f",
                "timestamp": "2025-12-26T01:26:41.177789",
                "agent": "Auditor_Agent",
                "model": "gemini-1.5-flash",
                "action": "CODE_ANALYSIS",
                "details": {
                    "input_prompt": "Analyze this code...",
                    "output_response": "I found 3 issues...",
                    "file_analyzed": "example.py"
                },
                "status": "SUCCESS"
            }
        }
