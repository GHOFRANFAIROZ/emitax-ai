# Yapay Zekâ Katmanı ile Blokzincir Arasındaki Teslim Sözleşmesi (Emitax)

AI katmanı her beyan için bütünlük parmak iziyle bir **doğrulama kaydı** üretir. Teslim iki düzeydedir:

## Zincire yazılan (on-chain) — `onchain.jsonl`
Yalnızca karar özeti + iki parmak izi (ham değer yok):

| Alan | Anlamı |
|---|---|
| `record_id` | `facility|unit|period` |
| `facility`, `unit`, `period` | tesis/birim/ay |
| `trust_score` | güven skoru 0–100 |
| `action` | `approve` / `request_docs` / `human_review` |
| `data_hash` | beyan edilen ve ölçülen değerlerin parmak izi |
| `record_hash` | tam kaydın parmak izi (çapa) |
| `verified_at`, `schema_version` | doğrulama zamanı ve sözleşme sürümü |

## Zincir dışında saklanan (off-chain) — `offchain.jsonl`
Tam kayıt: beyan edilen değerler, ölçüm özeti, tetiklenen kontroller ve okunabilir gerekçe.

## Bütünlük ilkesi (gizli anlaşmaya karşı)
- Değerler üzerinde sonradan yapılan her değişiklik `record_hash` değerini değiştirir → zincirdeki çapayla karşılaştırılarak tespit edilir.
- Karar ve kanıt (skor + gerekçeler + parmak izi) zincire **herhangi bir insan onayından bağımsız olarak** yazılır; insan denetçi yalnızca `human_review` üzerinde karar verir ve çapayı silemez.

## Programlama arayüzü
- `export_decisions(decisions, declarations, measurement, cfg) -> (onchain[], offchain[])`
- `verify_record(record) -> bool` (bir kaydın bütünlüğünü doğrulamak isteyen her taraf için).

Blokzincir ekibi `onchain.jsonl` dosyasını tüketir (`record_hash` + özeti sözleşmeye yazar) ve `offchain.jsonl` dosyasını zincir dışı bir veritabanında saklar.
