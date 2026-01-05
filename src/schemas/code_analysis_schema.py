from pydantic import BaseModel

class CodeAnalysisResult(BaseModel):
    file: str
    score: float
    issues: list[str]
    total_issues: int
    report: str