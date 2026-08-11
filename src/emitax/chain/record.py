from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone

# =========================================================
#  عقد الإخراج للبلوكتشين (الحلقة الأخيرة)
#  الـ AI يُنتج «سجل تحقّق» قانونياً مع بصمة سلامة (hash).
#  على السلسلة: الملخّص + البصمة (دليل ثابت مضادّ للتواطؤ).
#  خارج السلسلة: السجل الكامل (القيَم الخام).
#  تعديل أي قيمة يغيّر البصمة → يُكشف فوراً.
# =========================================================

SCHEMA_VERSION = "emitax-ai/1.0"


def _canonical(obj) -> str:
    """JSON قانوني حتمي (مفاتيح مرتّبة، بلا فراغات) — أساس بصمة قابلة لإعادة الإنتاج."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_record(decision: dict, declaration: dict, measurement: dict | None,
                 cfg: dict, verified_at: str | None = None) -> dict:
    """يبني سجل تحقّق كامل لبيان واحد.

    decision: صف من مخرجات fuse (trust_score, action, fired_checks, reason...).
    declaration: صف البيان (decl_<gas>...). measurement: صف القياس (اختياري).
    verified_at: مرّريه ثابتاً للاختبارات؛ افتراضياً وقت التحقّق الآن.
    """
    gases = list(cfg["columns"]["gases"])
    declared = {g: float(declaration[f"decl_{g}"]) for g in gases if f"decl_{g}" in declaration}
    measured = None
    if measurement is not None:
        measured = {g: float(measurement[f"{g}_mass"]) for g in gases if f"{g}_mass" in measurement}

    va = verified_at or datetime.now(timezone.utc).isoformat()
    rid = f"{decision['facility']}|{decision['unit']}|{decision['ym']}"
    data_hash = _sha256(_canonical({"declared": declared, "measured": measured}))

    fired = [c for c in str(decision.get("fired_checks", "")).split(",") if c]
    record = {
        "schema_version": cfg.get("chain", {}).get("schema_version", SCHEMA_VERSION),
        "record_id": rid,
        "facility": decision["facility"], "unit": decision["unit"], "period": decision["ym"],
        "declared": declared, "measured_summary": measured,
        "trust_score": int(decision["trust_score"]), "action": decision["action"],
        "fired_checks": fired, "reason": decision.get("reason", ""),
        "verified_at": va, "data_hash": data_hash,
    }
    # بصمة السجل تغطّي كل شيء عدا نفسها
    record["record_hash"] = _sha256(_canonical(record))
    return record


def onchain_payload(record: dict) -> dict:
    """ما يُسجَّل على السلسلة: ملخّص القرار + البصمتان فقط (بلا قيَم خام)."""
    return {
        "record_id": record["record_id"], "facility": record["facility"],
        "unit": record["unit"], "period": record["period"],
        "trust_score": record["trust_score"], "action": record["action"],
        "data_hash": record["data_hash"], "record_hash": record["record_hash"],
        "verified_at": record["verified_at"], "schema_version": record["schema_version"],
    }


def offchain_payload(record: dict) -> dict:
    """ما يُخزَّن خارج السلسلة: السجل الكامل (القيَم الخام + الأسباب)."""
    return dict(record)


def verify_record(record: dict) -> bool:
    """يتحقّق أنّ record_hash يطابق محتوى السجل (كشف أي تعديل لاحق)."""
    r = {k: v for k, v in record.items() if k != "record_hash"}
    return _sha256(_canonical(r)) == record.get("record_hash")


def export_decisions(decisions, declarations, measurement, cfg, verified_at: str | None = None):
    """يبني سجلّات التحقّق لكل القرارات ويُرجع (onchain[list], offchain[list]).

    يضمّ كل قرار بصفّي البيان والقياس المطابقين (على facility/unit/period).
    """
    keys = ["facility", "unit", "ym"]
    decl_idx = {(r["facility"], r["unit"], r["ym"]): r for _, r in declarations.iterrows()}
    meas_idx = {(r["facility"], r["unit"], r["ym"]): r for _, r in measurement.iterrows()} if measurement is not None else {}

    onchain, offchain = [], []
    for _, d in decisions.iterrows():
        k = (d["facility"], d["unit"], d["ym"])
        decl = decl_idx.get(k)
        if decl is None:
            continue
        rec = build_record(d.to_dict(), decl.to_dict(),
                           meas_idx.get(k).to_dict() if k in meas_idx else None,
                           cfg, verified_at)
        onchain.append(onchain_payload(rec))
        offchain.append(offchain_payload(rec))
    return onchain, offchain