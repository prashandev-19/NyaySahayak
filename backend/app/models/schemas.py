from pydantic import BaseModel, Field
from typing import List, Optional


class CaseAnalysisResponse(BaseModel):
    case_id: str
    summary: str
    offenses: List[str]         
    sections: List[str] = Field(default_factory=list)
    missing_evidence: List[str] 
    recommendation: str
    summary_hindi: Optional[str] = None
    offenses_hindi: Optional[List[str]] = None
    missing_evidence_hindi: Optional[List[str]] = None