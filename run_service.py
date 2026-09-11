"""
Emitax AI Doğrulama Servisini başlatır.

    python run_service.py

Ardından tarayıcıdan:  http://127.0.0.1:8300/docs

Bu servis TEK BAŞINA çalışır; kardeş klasörlere bağımlı değildir. Dört kontrolü
koşar ve güven skorunu üretir; veritabanı, zincir ve arayüz bilmez.

Artefaktları (aylık ölçüm referansı, saatlik parquet, ONNX modeli) kendi
`data/processed/` klasöründen okur. Eksik olan artefaktın kontrolü "kullanılamaz"
işaretlenir; servis çökmez, /health durumu bildirir.

Çekirdek servis buraya EMITAX_AI_URL ile bağlanır (varsayılan 8300).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import uvicorn                       # noqa: E402
from emitax.config import load_config  # noqa: E402


def main() -> None:
    # Yapılandırma MUTLAK yolla okunur: servis hangi dizinden başlatılırsa
    # başlatılsın aynı ayarları ve aynı artefaktları kullanır.
    try:
        svc = (load_config(os.path.join(ROOT, "configs", "default.yaml"))
               or {}).get("service", {}) or {}
    except Exception:
        svc = {}
    host = os.getenv("EMITAX_AI_HOST", svc.get("host", "127.0.0.1"))
    port = int(os.getenv("EMITAX_AI_PORT", svc.get("port", 8300)))
    print(f"Emitax AI doğrulama servisi:  http://{host}:{port}/docs")
    uvicorn.run("emitax.service.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
