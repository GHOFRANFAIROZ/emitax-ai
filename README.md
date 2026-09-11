# Emitax — AI Verification Layer

Emitax projesi için veri denetim (yapay zekâ) katmanı — TEKNOFEST 2026 / BlockSec26.

## Çalıştırma (yerel — VSCode)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# CEMS dosyasını data/raw/ içine koyun, ardından:
python run_prepare.py     # yükleme + aylık toplulaştırma + çıktıları kaydet
python run_eda.py         # docs/eda/ içine EDA görselleri
pytest -q                 # testler
```

## Dosyalar
- `configs/default.yaml` — tüm ayarlar (sütun adları, gazlar, toplulaştırma dönemi). Yalnızca veri değişirse burayı düzenleyin.
- `src/emitax/data/loader.py` — sütun/boyut bağımsız CEMS yükleme (parça parça okumayı destekler).
- `src/emitax/data/aggregate.py` — aylık toplulaştırma (= Kontrol 1 referans ölçümü) + ölçüm mevcudiyeti tespiti.
- `run_prepare.py` / `run_eda.py` — Aşama 1 betikleri.

## Esneklik (Scalability)
Her şey ayar-güdümlü; yükleyici fazla sütunları yok sayar ve herhangi bir sayıda birim/tesisle çalışır.
Çok büyük dosyalar için: `data.chunksize` değerini ayarlayın.

---

## Tek başına çalıştırma

Bu servis kardeş klasörlere **bağımlı değildir**. Artefaktlarını kendi
`data/processed/` klasöründen okur ve hangi dizinden başlatılırsa başlatılsın
aynı dosyaları bulur.

```bash
pip install -r requirements.txt        # ya da: pip install -e .
python run_service.py                  # -> http://127.0.0.1:8300/docs
```

Ayarlar: `.env.example` dosyasına bakın (`EMITAX_AI_HOST`, `EMITAX_AI_PORT`).

### Uçlar

| Uç | Ne yapar |
|---|---|
| `GET /health` | Dört kontrolün hangisi aktif, kaç referans kayıt yüklü |
| `POST /verify` | Bir beyanı doğrular -> skor, tetiklenen kontroller, iki hash |
| `POST /verify/batch` | Beyan listesini doğrular |
| `GET /reference` | Aylık ölçüm referansı (demo kurulumu bunu kullanır) |

### Artefaktı eksikse

Motor **çökmez**. Eksik artefaktın kontrolü "kullanılamaz" işaretlenir ve
`/health` kaçının aktif olduğunu bildirir. Örneğin `lstm_ae.onnx` yoksa
zamansal kontrol atlanır, diğer üçü çalışır.

### Testler

```bash
python -m pytest tests/ -q
```


---

## Emitax'taki yeri

Emitax altı bağımsız parçadan oluşur; her biri kendi klasöründe tek başına
kurulur, kendi testlerini koşar ve kendi süreci olarak çalışır.

| Port | Parça | Klasör |
|---|---|---|
| :8400 | Zincir Geçidi | `emitax-network` |
| :8300 | AI Doğrulama | `emitax-ai` |
| :8100 | Denetçi Sicili | `emitax-denetci/emitax-denetci-backend` |
| :8000 | Çekirdek Servis | `emitax-arayuz/emitax-arayuz-backend` |
| :8200 | Denetçi Portalı | `emitax-denetci/emitax-denetci-frontend` |
| — | Masaüstü | `emitax-arayuz/emitax-arayuz-frontend` |

Parçalar birbirine **yalnızca HTTP ile** bağlanır — dosya sistemi ya da
`import` üzerinden değil. Hepsini tek komutla kaldırmak için kök klasörde:
`.\baslat.ps1`
