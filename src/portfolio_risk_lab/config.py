"""经校验的分析参数。"""

from pydantic import BaseModel, ConfigDict, Field


class AnalysisConfig(BaseModel):
    """全局分析约定。"""

    model_config = ConfigDict(frozen=True)

    periods_per_year: int = Field(default=252, ge=2, le=366)
    annual_risk_free_rate: float = Field(default=0.0, gt=-1.0, lt=1.0)
    var_confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
