"""Pydantic request/response models for the Emitax verification service."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DeclarationIn(BaseModel):
    """One monthly facility declaration (beyan) to be audited."""
    facility_id: str = Field(..., description="Tesis kimliği")
    unit_id: str = Field(..., description="Ünite kimliği (ör. LIM1)")
    year: int = Field(..., ge=2000, le=2100)
    month: int = Field(..., ge=1, le=12)

    # energy proxy — required for physics/peer intensity
    heat_input: float = Field(..., description="Aylık toplam ısı girdisi (enerji)")
    fuel_type: str = Field("Coal", description="Yakıt türü (IPCC faktörü seçimi)")

    # declared emissions (aylık toplam kütle)
    co2: float = Field(..., description="Beyan edilen CO2")
    so2: Optional[float] = Field(None, description="Beyan edilen SO2")
    nox: Optional[float] = Field(None, description="Beyan edilen NOx")
    pm: Optional[float] = Field(None, description="Beyan edilen PM (ölçümde yok)")

    # optional descriptive fields (kept for the record / UI)
    fuel_amount: Optional[float] = None
    electricity: Optional[float] = None
    production: Optional[float] = None


class CheckDetail(BaseModel):
    flagged: Optional[bool] = None          # None = check not applicable
    detail: Optional[str] = None


class VerifyOut(BaseModel):
    facility_id: str
    unit_id: str
    ym: str                                 # "YYYY-MM"
    trust_score: float                      # 0..100
    decision: str                           # approve | request_docs | human_review
    fired_checks: List[str]
    reason: str
    per_check: Dict[str, CheckDetail]
    record_hash: str
    onchain_payload: Dict[str, Any]
