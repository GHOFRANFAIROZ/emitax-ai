# Emitax Doğrulama Servisi — API Sözleşmesi (Faz A)

REST servisi doğrulama motorunu sarmalar. Masaüstü arayüz (PySide6) ve
blokzincir yazıcısı bu sözleşme üzerinden entegre olur.

## Çalıştırma
```
pip install -r requirements.txt
python run_service.py
# Dokümantasyon:  http://127.0.0.1:8000/docs
```

## GET /health
Servis durumu ve aktif kontroller.
```json
{ "status": "ok", "engine": true,
  "checks_available": {"measurement": true, "physics": true, "peer": true, "temporal": true},
  "reference_records": 24, "temporal_flags": 24 }
```

## POST /verify
Bir aylık beyanı denetler.

İstek (DeclarationIn):
```json
{ "facility_id": "Limestone", "unit_id": "LIM1",
  "year": 2025, "month": 6,
  "heat_input": 5000000, "fuel_type": "Coal",
  "co2": 525000, "so2": 1200, "nox": 900, "pm": 50 }
```

Yanıt (VerifyOut):
```json
{ "facility_id": "Limestone", "unit_id": "LIM1", "ym": "2025-06",
  "trust_score": 100.0, "decision": "approve",
  "fired_checks": [],
  "reason": "Beyan tüm kontrolleri geçti.",
  "per_check": {
    "measurement": {"flagged": false, "detail": "..."},
    "physics":     {"flagged": false, "detail": "..."},
    "peer":        {"flagged": false, "detail": "..."},
    "temporal":    {"flagged": false, "detail": "..."} },
  "record_hash": "…",
  "onchain_payload": { "schema_version": "emitax-ai/1.0", "…": "…" } }
```

- `decision` ∈ `approve` (≥85) | `request_docs` (55–85) | `human_review` (<85 alt).
- `fired_checks`: dile bağımsız anahtarlar — Türkçe arayüz bunları kendi
  etiketlerine çevirir.
- `onchain_payload`: blokzincire yazılacak **özet + hash** (ham beyan değerleri
  yok). Blokzincir servisi bu payload'ı Solidity kayıt sözleşmesine iletir.

## POST /verify/batch
Beyan listesi alır, `VerifyOut` listesi döner.

## Notlar
- Motor, `run_prepare.py` + Colab dışa aktarımının ürettiği artefaktları okur:
  `monthly_measurement.csv`, `hourly_clean.parquet`, `lstm_ae.onnx`,
  `lstm_meta.json` (yollar `configs/default.yaml → service` altında).
- Bir kontrolün artefaktı yoksa o kontrol `null` döner; skor kalan
  kontrollerden hesaplanır (servis ayakta kalır).
