import subprocess
import re
import os
from src.schemas.code_analysis_schema import CodeAnalysisResult


class StaticAnalyzerService:
    
    def __init__(self, ignore_style_issues=False):
        """
        Args:
            ignore_style_issues: If True, filters out minor style issues
        """
        self.ignore_style_issues = ignore_style_issues
        self.ignore_keywords = [
            'missing-final-newline',
            'invalid-name',
            'line-too-long',
            'too-few-public-methods',
            'Final newline missing',
            'Missing module docstring',
            'UPPER_CASE naming style',

        ]
    
    def run_pylint(self, file_path: str) -> tuple:
        """
        Run pylint on a file and return score + report.
        
        Returns:
            (score, report) - score is 0-10, report is the full text
        """
        # Validate file exists
        if not os.path.exists(file_path):
            return 0.0, f"Error: File '{file_path}' not found"
        
        try:
            result = subprocess.run(
                [
                    'pylint',
                    file_path,
                    '--score=yes',
                    '--msg-template={path}:{line}:{column}: {msg}'
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout
            
            # Extract score: "rated at 7.50/10"
            match = re.search(r'rated at ([\-\d\.]+)/10', output)
            score = float(match.group(1)) if match else 0.0
            score = max(0.0, score)  # No negative scores
            
            return score, output
        
        except subprocess.TimeoutExpired:
            return 0.0, "Error: Pylint timed out"
        except FileNotFoundError:
            return 0.0, "Error: Pylint not installed. Run: pip install pylint"
        except Exception as e:
            return 0.0, f"Error: {str(e)}"
    
    def get_issues(self, pylint_report: str, important_only: bool = True) -> list:
        """
        Extract issues from pylint output.
        
        Returns:
            List of strings like "Line 5: Missing docstring"
        """
        issues = []
        
        # Match lines like: "file.py:10:0: Missing docstring"
        pattern = r':(\d+):\d+:\s*(.+)'
        
        for match in re.finditer(pattern, pylint_report):
            line_num = match.group(1)
            message = match.group(2).strip()
            
            issue_str = f"Line {line_num}: {message}"
            issues.append(issue_str)
        
        # Auto-filter if configured
        if important_only:
            issues = self._filter_issues(issues)
        
        return issues
    
    def _filter_issues(self, issues: list) -> list:
        """Filter out minor style issues."""
        important = []
        for issue in issues:
            if not any(keyword in issue for keyword in self.ignore_keywords):
                important.append(issue)
        return important
    def _format_readable(self, file_path: str, score: float, issues: list) -> str:
        """Format issues cleanly for the Fixer Agent."""
        report = f"""CODE QUALITY REPORT
        File: {os.path.basename(file_path)}
        Score: {score}/10

        ISSUES TO FIX:
        """
        if issues:
            for i, issue in enumerate(issues, 1):
                report += f"{i}. {issue}\n"
        else:
            report += "No issues found!\n"
        
        return report.strip()
    
    def analyze(self, file_path: str, important_only: bool = True) -> CodeAnalysisResult:
        """
        Complete analysis of a Python file using pylint.
        
        Args:
            file_path: Path to the Python file to analyze.
            important_only: If True, filters out minor style issues.
        
        Returns:
            CodeAnalysisResult: A Pydantic model containing:
                - file (str): Name of the analyzed file
                - score (float): Pylint score (0-10)
                - issues (List[str]): List of issues found
                - total_issues (int): Number of issues
                - report (str): Full pylint output
        """

        score, report = self.run_pylint(file_path)
        issues = self.get_issues(report, important_only)
        
        return CodeAnalysisResult(
            file=os.path.basename(file_path),
            score=score,
            issues=issues,
            total_issues=len(issues),
            report=report
        )


#instantiation
static_analyzer_service = StaticAnalyzerService(ignore_style_issues=True)
