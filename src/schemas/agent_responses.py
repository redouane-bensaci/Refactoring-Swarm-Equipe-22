from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AuditorResponse:
    """
    Response schema for Auditor Agent.
    Used with LangChain's ToolStrategy for structured output.
    
    Fields:
        summary: Summary of the analysis
        files_analyzed: List of file paths that were analyzed
        refactoring_plan: List of refactoring items (prioritized issues to fix)
        overall_score: Overall code quality score (0-10)
        critical_issues: Number of critical issues found
    """
    summary: str
    files_analyzed: List[str] = field(default_factory=list)
    refactoring_plan: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    critical_issues: int = 0


@dataclass
class FixerResponse:
    """
    Response schema for Fixer Agent.
    Used with LangChain's ToolStrategy for structured output.
    
    Fields:
        success: Whether the fix was successful
        file_fixed: Path to the file that was fixed
        changes_made: List of descriptions of changes made
        error: Error message if fix failed (optional)
    """
    success: bool
    file_fixed: str
    changes_made: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class JudgeResponse:
    """
    Response schema for Judge Agent.
    Used with LangChain's ToolStrategy for structured output.
    
    Fields:
        all_tests_passed: Whether all tests passed
        tests_passed: Number of tests that passed
        tests_failed: Number of tests that failed
        error_logs: Error logs if tests failed (optional)
        quality_score: Code quality score after fixes (optional)
    """
    all_tests_passed: bool
    tests_passed: int = 0
    tests_failed: int = 0
    error_logs: Optional[str] = None
    quality_score: Optional[float] = None
