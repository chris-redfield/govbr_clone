from pydantic import BaseModel, Field
from typing import Dict

class CriteriaResponse(BaseModel):
    crit_1: int = Field(description="Criteria 1 score")
    crit_2: int = Field(description="Criteria 2 score")
    crit_3: int = Field(description="Criteria 3 score")
    crit_4: int = Field(description="Criteria 4 score")
    crit_5: int = Field(description="Criteria 5 score")
    crit_6: int = Field(description="Criteria 6 score")
    crit_7: int = Field(description="Criteria 7 score")
    crit_8: int = Field(description="Criteria 8 score")
