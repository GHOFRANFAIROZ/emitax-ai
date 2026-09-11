"""RAG showcase -- ekip demosu (VEKTOR DB / embeddings, gercek resmi mevzuat).

MIMARI: RAG kontrole bagli DEGILDIR; her ay BUTUN tesislere calisir.
- Temiz kayit  -> yasal UYGUNLUK teyidi
- Supheli kayit -> yasal DAYANAK / aykirilik onerisi
Kontrol sonuclari, sorguya EK girdi olarak katilir.

ILK CALISMA: model indirilir (~470 MB, internet) + vektorler hesaplanir (birkac
dakika), data/legal_corpus/.embed_index/ altina kaydedilir. Sonraki calismalar
cache'ten hizli yuklenir.
    pip install sentence-transformers faiss-cpu
    python run_rag_showcase.py
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

TARGETS = [
    ("Limestone", "LIM1", "2025-01", "TEMIZ kayit (kontrol yok) -> uygunluk teyidi"),
    ("Limestone", "LIM1", "2025-06", "SUPHELI: olcum -> SKHKKY Ek-5"),
    ("Limestone", "LIM1", "2025-12", "SUPHELI: dort kontrol -> coklu kaynak"),
]


def _find(df, f, u, y):
    m = (df["facility"] == f) & (df["unit"] == u) & (df["ym"].astype(str) == y)
    sub = df[m]
    return sub.iloc[0].to_dict() if len(sub) else None


def main() -> None:
    try:
        import pandas as pd
    except Exception:
        print("pandas gerekli."); return
    if not CSV.exists():
        print(f"Bulunamadi: {CSV}"); return
    print("Vektor indeksi hazirlaniyor (ilk calisma birkac dakika surebilir)...")
    index = get_embed_index(CORPUS)
    print(f"Indeks hazir: {len(index)} parca | model: {index.model_name}")
    print("MIMARI: RAG her ay butun tesislere calisir (kontrole bagli degil).\n")
    df = pd.read_csv(CSV)
    for f, u, y, desc in TARGETS:
        row = _find(df, f, u, y)
        if row is None:
            print(f"(atlandi: {f}/{u}/{y})\n"); continue
        print("#" * 72)
        print(f"# {desc}   [{f}/{u}/{y}]")
        print("#" * 72)
        print(recommend(index, row, k=2).text)
        print()


if __name__ == "__main__":
    main()
