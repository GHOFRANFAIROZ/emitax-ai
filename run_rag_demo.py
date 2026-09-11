"""RAG demo (VEKTOR DB / embeddings, gercek resmi mevzuat).

    pip install sentence-transformers faiss-cpu
    python run_rag_demo.py
Ilk calisma modeli indirir + vektorleri hesaplar; sonra cache'ten hizli yuklenir.
"""
from __future__ import annotations
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from emitax.rag.embed_index import get_embed_index    # noqa: E402
from emitax.rag.recommend import recommend            # noqa: E402

CORPUS = ROOT / "data" / "legal_corpus"
CSV = ROOT / "data" / "processed" / "trust_decisions.csv"
FALLBACK = {"facility": "ORNEK", "unit": "U1", "ym": "2025-06", "trust_score": 55,
            "action": "request_docs", "fired_checks": "measurement",
            "reason": "Beyan olcumun altinda"}


def main() -> None:
    print("Vektor indeksi hazirlaniyor (ilk calisma birkac dakika)...")
    index = get_embed_index(CORPUS)
    print(f"Indeks hazir: {len(index)} parca | model: {index.model_name}\n")
    record, origin = FALLBACK, "ornek kayit"
    try:
        import pandas as pd
        if CSV.exists():
            df = pd.read_csv(CSV)
            m = df["is_tampered"].astype(str).isin(["1", "True", "true"]) if "is_tampered" in df else None
            row = df[m].iloc[0] if (m is not None and m.any()) else df.iloc[0]
            record, origin = row.to_dict(), f"{CSV.name} icinden"
    except Exception:
        pass
    print(f"Secilen kayit: {origin}\n")
    print(recommend(index, record, k=3).text)


if __name__ == "__main__":
    main()
