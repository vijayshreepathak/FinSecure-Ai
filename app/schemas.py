from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    Time: float
    Amount: float
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Class: Optional[int] = Field(default=None)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictionOut(BaseModel):
    probability: float
    risk_score: float
    label: int
    threshold: float
    iso_score_raw: float
    iso_norm: float
    ae_error: float
    ae_norm: float
    explanations: List[FeatureContribution]


class BatchPredictionOut(BaseModel):
    results: List[PredictionOut]
    count: int


class MetricsOut(BaseModel):
    total: int
    fraud_count: int
    fraud_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    threshold: float
    by_hour: List[Dict[str, Any]]
    amount_hist: List[Dict[str, Any]]


class AlertOut(BaseModel):
    id: int
    created_at: str
    amount: float
    risk_score: float
    probability: float
    threshold: float


