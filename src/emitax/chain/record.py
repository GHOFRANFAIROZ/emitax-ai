from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone

# =========================================================
#  Blokzincir çıkış sözleşmesi (son halka)
#  AI, bütünlük parmak izi (hash) ile hukuki bir «doğrulama kaydı» üretir.
#  Zincirde: özet + parmak izi (değişmez, gizli anlaşmaya karşı kanıt).
#  Zincir dışında: tam kayıt (ham değerler).
#  Herhangi bir değeri değiştirmek parmak izini değiştirir → anında tespit edilir.
# =========================================================

SCHEMA_VERSION = "emitax-ai/1.0"


def _canonical(obj) -> str:
    """Belirlenimci kanonik JSON (sıralı anahtarlar, boşluksuz) — tekrarlanabilir parmak izinin temeli."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_record(decision: dict, declaration: dict, measurement: dict | None,
                 cfg: dict, verified_at: str | None = None) -> dict:
    """Tek bir beyan için eksiksiz doğrulama kaydı oluşturur.

    decision: fuse çıktısından bir satır (trust_score, action, fired_checks, reason...).
    declaration: beyan satırı (decl_<gas>...). measurement: ölçüm satırı (opsiyonel).
    verified_at: testler için sabit verin; varsayılan olarak şu anki doğrulama zamanı.
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
    # Kayıt parmak izi, kendisi hariç her şeyi kapsar
    record["record_hash"] = _sha256(_canonical(record))
    return record


def onchain_payload(record: dict) -> dict:
    """Zincire yazılan: karar özeti + iki parmak izi (ham değer yok)."""
    return {
        "record_id": record["record_id"], "facility": record["facility"],
        "unit": record["unit"], "period": record["period"],
        "trust_score": record["trust_score"], "action": record["action"],
        "data_hash": record["data_hash"], "record_hash": record["record_hash"],
        "verified_at": record["verified_at"], "schema_version": record["schema_version"],
    }


def offchain_payload(record: dict) -> dict:
    """Zincir dışında saklanan: tam kayıt (ham değerler + gerekçeler)."""
    return dict(record)


def verify_record(record: dict) -> bool:
    """record_hash'in kayıt içeriğiyle eşleştiğini doğrular (sonraki her değişikliği tespit)."""
    r = {k: v for k, v in record.items() if k != "record_hash"}
    return _sha256(_canonical(r)) == record.get("record_hash")


def export_decisions(decisions, declarations, measurement, cfg, verified_at: str | None = None):
    """Tüm kararlar için doğrulama kayıtları oluşturur ve (onchain[list], offchain[list]) döndürür.

    Her kararı, eşleşen beyan ve ölçüm satırlarıyla birleştirir (facility/unit/period üzerinden).
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