"""
Pylint Tool - Wrapper for running pylint static analysis
"""
import subprocess
import json
from typing import Dict, Any, List


def run_pylint_analysis(file_path: str) -> Dict[str, Any]:
    """
    Run pylint analysis on a Python file
    
    Args:
        file_path: Path to Python file to analyze
        
    Returns:
        Dictionary with pylint results:
        - score: Overall code quality score (0-10)
        - issues: List of detected issues
        - raw_output: Raw pylint output
    """
    try:
        # Run pylint with JSON output format
        result = subprocess.run(
            ['pylint', file_path, '--output-format=json', '--score=yes'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse JSON output
        try:
            issues = json.loads(result.stdout) if result.stdout else []
        except json.JSONDecodeError:
            issues = []
        
        # Extract score from stderr (pylint outputs score there)
        score = _extract_score(result.stderr)
        
        return {
            "score": score,
            "issues": issues,
            "total_issues": len(issues),
            "raw_output": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "score": 0,
            "issues": [],
            "total_issues": 0,
            "error": "Pylint timeout (>30s)"
        }
    except FileNotFoundError:
        return {
            "score": 0,
            "issues": [],
            "total_issues": 0,
            "error": "Pylint not installed. Run: pip install pylint"
        }
    except Exception as e:
        return {
            "score": 0,
            "issues": [],
            "total_issues": 0,
            "error": f"Pylint execution failed: {str(e)}"
        }


def _extract_score(stderr_output: str) -> float:
    """Extract pylint score from stderr output"""
    if not stderr_output:
        return 0.0
    
    # Look for "Your code has been rated at X.XX/10"
    for line in stderr_output.split('\n'):
        if "rated at" in line.lower():
            try:
                # Extract number before "/10"
                parts = line.split("rated at")[1].split("/10")[0]
                score = float(parts.strip())
                return score
            except (IndexError, ValueError):
                pass
    
    return 0.0


def get_issue_summary(pylint_result: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of pylint issues
    
    Args:
        pylint_result: Result from run_pylint_analysis()
        
    Returns:
        Formatted string summary
    """
    if "error" in pylint_result:
        return f"Error: {pylint_result['error']}"
    
    score = pylint_result.get("score", 0)
    total = pylint_result.get("total_issues", 0)
    
    if total == 0:
        return f"✅ Clean code! Score: {score:.2f}/10"
    
    # Categorize issues
    issues_by_type = {}
    for issue in pylint_result.get("issues", []):
        issue_type = issue.get("type", "unknown")
        issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
    
    summary = f"Score: {score:.2f}/10 | Total issues: {total}\n"
    for issue_type, count in issues_by_type.items():
        summary += f"  - {issue_type}: {count}\n"
    
    return summary.strip()


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pylint_tool.py <path_to_python_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    result = run_pylint_analysis(file_path)
    
    print("="*60)
    print(f"Pylint Analysis: {file_path}")
    print("="*60)
    print(get_issue_summary(result))
    print("\nTop 5 Issues:")
    for i, issue in enumerate(result.get("issues", [])[:5], 1):
        print(f"{i}. Line {issue.get('line', '?')}: {issue.get('message', 'Unknown')}")