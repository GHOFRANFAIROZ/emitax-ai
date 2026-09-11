"""RAG guven esigi (MIN_SCORE) FINAL DOGRULAMA.

    python run_rag_final_check.py

run_rag_calibration.py ile MIN_SCORE'u recommend.py'ye yazdiktan SONRA calistirin.
Uc senaryo test edilir:
  A) SUPHELI + gecerli bulgu  -> kaynak ALINTILANMALI (benzerlik esigin ustunde).
  B) TEMIZ kayit              -> UYGUNLUK TEYIDI uretilmeli.
  C) Zorlanmis yuksek esik    -> guard devreye girmeli: kaynak yerine
                                 "Yeterli guven duzeyinde mevzuat dayanagi bulunamadi."
Her senaryo icin PASS/FAIL basilir. C, guard kablolamasini deterministik dogrular.
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
from emitax.rag.embed_index import get_embed_index                       # noqa: E402
from emitax.rag.recommend import recommend, MIN_SCORE, NO_BASIS_MSG      # noqa: E402

CORPUS = ROOT / "data" / "legal_corpus"

REC_SUSPECT = {"facility": "ORNEK", "unit": "U1", "ym": "2025-06", "trust_score": 58,
               "action": "request_docs", "fired_checks": "physics",
               "reason": "Yakit tuketimine gore CO2 fiziksel olarak tutarsiz"}
REC_CLEAN = {"facility": "ORNEK", "unit": "U1", "ym": "2025-03", "trust_score": 95,
             "action": "approve", "fired_checks": "", "reason": ""}


def _has_citation(text: str) -> bool:
    return "benzerlik:" in text


def _has_refusal(text: str) -> bool:
    return NO_BASIS_MSG in text


def main() -> None:
    print("Vektor indeksi hazirlaniyor (ilk calisma birkac dakika surebilir)...")
    index = get_embed_index(CORPUS)
    print(f"Indeks hazir: {len(index)} parca | model: {index.model_name}")
    print(f"Aktif MIN_SCORE (recommend.py): {MIN_SCORE:.2f}\n")

    results = []

    # --- A) SUPHELI + gecerli bulgu -> kaynak alintilanmali ---
    a = recommend(index, REC_SUSPECT, k=2)
    a_pass = _has_citation(a.text) and not _has_refusal(a.text)
    results.append(("A: supheli kayit kaynak alintiliyor", a_pass))
    print("#" * 72); print("# A) SUPHELI + gecerli bulgu (esik: aktif MIN_SCORE)")
    print("#" * 72); print(a.text); print()

    # --- B) TEMIZ kayit -> uygunluk teyidi ---
    b = recommend(index, REC_CLEAN, k=2)
    b_pass = "UYGUNLUK TEYIDI" in b.text
    results.append(("B: temiz kayit uygunluk teyidi", b_pass))
    print("#" * 72); print("# B) TEMIZ kayit -> UYGUNLUK TEYIDI")
    print("#" * 72); print(b.text); print()

    # --- C) Zorlanmis yuksek esik -> guard devreye girmeli ---
    c = recommend(index, REC_SUSPECT, k=2, min_score=0.99)
    c_pass = _has_refusal(c.text) and not _has_citation(c.text)
    results.append(("C: yuksek esikte guard reddediyor", c_pass))
    print("#" * 72); print("# C) Zorlanmis esik=0.99 -> guard REDDETMELI")
    print("#" * 72); print(c.text); print()

    # --- ozet ---
    print("=" * 72); print("SONUC"); print("=" * 72)
    all_ok = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
        all_ok = all_ok and ok
    print("=" * 72)
    print("TUM TESTLER GECTI -- RAG guard hazir." if all_ok
          else "BAZI TESTLER BASARISIZ -- yukaridaki ciktiyi inceleyin.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
