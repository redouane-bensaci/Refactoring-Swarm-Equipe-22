"""
LangChain tools for code analysis and testing.
These tools wrap static_analyzer service and provide test execution capabilities.
"""

import os
import subprocess
from langchain.tools import tool, ToolRuntime

from src.services.static_analyzer import static_analyzer_service
from src.tools.file_tools import CodeContext  # Import shared context


@tool
def analyze_code_quality(file_path: str, runtime: ToolRuntime[CodeContext]) -> str:
    """Run pylint static analysis on a Python file to check code quality.
    
    Use this tool to get a quality score and list of issues for a Python file.
    This helps identify bugs, code smells, and quality issues.
    
    Args:
        file_path: Path to the file to analyze (relative to target_dir or absolute)
    
    Returns:
        Formatted analysis report with score (0-10), total issues, and list of issues
    """
    # Resolve path
    if not os.path.isabs(file_path):
        file_path = os.path.join(runtime.context.target_dir, file_path)
    
    result = static_analyzer_service.analyze(file_path)
    
    report = f"""Code Quality Analysis for {result.file}
Score: {result.score}/10
Total Issues: {result.total_issues}

Issues:
"""
    if result.issues:
        for i, issue in enumerate(result.issues[:15], 1):  # Limit to first 15 issues
            report += f"  {i}. {issue}\n"
    else:
        report += "  No issues found!\n"
    
    return report.strip()


@tool
def run_tests(target_dir: str, runtime: ToolRuntime[CodeContext]) -> str:
    """Run pytest on the target directory to validate code with unit tests.
    
    Use this tool to execute all unit tests and check if the code works correctly.
    This is critical for validating that fixes didn't break functionality.
    
    Args:
        target_dir: Directory to run tests on (usually same as context.target_dir)
    
    Returns:
        Test results summary with counts of passed/failed tests and error details
    """
    try:
        result = subprocess.run(
            ["pytest", target_dir, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        # Count test results
        passed = output.count("PASSED")
        failed = output.count("FAILED")
        error = output.count("ERROR")
        
        summary = f"""Test Results:
        Passed: {passed}
        Failed: {failed}
        Errors: {error}
        Return Code: {result.returncode}

        Full Output:
        {output[:2000]}  # Limit output size to avoid token overflow
        """
        return summary
    except subprocess.TimeoutExpired:
        return "Error: Tests timed out after 60 seconds"
    except FileNotFoundError:
        return "Error: pytest not found. Make sure pytest is installed."
    except Exception as e:
        return f"Error running tests: {str(e)}"
