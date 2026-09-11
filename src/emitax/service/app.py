"""
Emitax verification REST service (FastAPI).

Endpoints
  GET  /health          -> service status + which checks are active
  POST /verify          -> audit one declaration
  POST /verify/batch    -> audit a list of declarations

Run:  python run_service.py     (or)   uvicorn emitax.service.app:app --reload
Docs: http://127.0.0.1:8000/docs   (auto OpenAPI — teslim paketine dahil)
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from emitax.service.schemas import DeclarationIn, VerifyOut
from emitax.service.engine import build_engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("emitax.service")

app = FastAPI(
    title="Emitax — Doğrulama Servisi",
    description="Beyan doğrulama motoru (4 kontrol + güven skoru). AI katmanı / BlockSec26.",
    version="1.0.0",
)

# Desktop UI (PySide6) ve blokzincir yazıcısı bu servisi çağırır — CORS açık.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return _engine


@app.on_event("startup")
def _startup():
    global _engine
    try:
        _engine = build_engine()
        log.info("engine loaded; checks=%s", _engine.checks_available)
    except Exception as e:  # keep the service up so /health can report the problem
        log.error("engine load failed: %s", e)
        _engine = None


@app.get("/health")
def health():
    if _engine is None:
        return {"status": "degraded", "engine": False}
    return {
        "status": "ok",
        "engine": True,
        "checks_available": _engine.checks_available,
        "reference_records": len(_engine.meas_index),
        "temporal_flags": len(_engine.temporal_flags),
    }


@app.get("/reference")
def reference(facility: str | None = None, limit: int = 0):
    """Aylık ölçüm referansı (measurement kontrolünün karşılaştırma tabanı).

    Demo kurulum betikleri dürüst beyanı bu tablodan üretir. Artefakt dosyasını
    kardeş klasörden okumak yerine buradan istemeleri, AI katmanının dosya
    düzenini dışarıya sızdırmamasını sağlar.

    Ham ölçüm DEĞİLDİR; saatlik seriden toplanmış aylık özettir.
    """
    eng = get_engine()
    if eng.measurement is None:
        raise HTTPException(status_code=404, detail="ölçüm referansı yüklenmedi")
    df = eng.measurement
    if facility:
        col = eng.meas_cols.get("facility")
        if col:
            df = df[df[col].astype(str) == facility]
    if limit and limit > 0:
        df = df.head(limit)
    return {"columns": list(df.columns), "count": int(len(df)),
            "rows": df.to_dict(orient="records")}


@app.post("/verify", response_model=VerifyOut)
def verify(declaration: DeclarationIn):
    return get_engine().verify(declaration)


@app.post("/verify/batch", response_model=List[VerifyOut])
def verify_batch(declarations: List[DeclarationIn]):
    eng = get_engine()
    return [eng.verify(d) for d in declarations]


@app.get("/")
def root():
    return {"service": "emitax-verification", "docs": "/docs", "health": "/health"}
