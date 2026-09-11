"""RAG guven esigi (MIN_SCORE) KALIBRASYONU -- routing sonrasi.

    python run_rag_calibration.py

NE YAPAR:
- Vektor indeksini yukler (data/legal_corpus/.embed_index).
- VALID sorgular: her biri gercek bir yasal kaynaga karsilik gelmesi GEREKEN,
  uretimdeki gibi kontrol-yonlendirmeli (grup boost) sorgular. En iyi benzerlik
  skorlari toplanir -> MIN VALID (en zayif dogru eslesme).
- INVALID sorgular: hicbir yasal kaynaga karsilik gelmemesi GEREKEN, alan-disi
  sorgular. Her biri DORT rotanin hepsinden gecirilir, en yuksek skor alinir
  (en kotu durum sizinti) -> MAX INVALID (en guclu yanlis eslesme).
- Skorlar retrieve() ciktisidir: ROUTING SONRASI kosinus benzerligi (ham FAISS
  degil). Sektor/grup filtresi + boost zaten uygulanmistir.

CIKTI: MIN VALID, MAX INVALID, aralik (gap) ve onerilen MIN_SCORE.
Bu degeri src/emitax/rag/recommend.py icindeki MIN_SCORE'a yazin, sonra
run_rag_final_check.py ile dogrulayin.
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
from emitax.rag.embed_index import get_embed_index                      # noqa: E402
from emitax.rag.retrieve import retrieve                                # noqa: E402
from emitax.rag.recommend import (CHECK_QUERY_MAP, CHECK_GROUP_PREF,    # noqa: E402
                                  _BOOST, MIN_SCORE)

CORPUS = ROOT / "data" / "legal_corpus"

# --- VALID: gercek bir kaynaga karsilik gelmesi GEREKEN (kontrol, sorgu) ------
# Uretimdeki gibi: her sorgu kendi kontrolunun grup boost'u ile yonlendirilir.
VALID = [
    ("measurement", CHECK_QUERY_MAP["measurement"]),
    ("measurement", "baca gazi emisyon sinir degeri asildi mg/Nm3 beyan olcum altinda"),
    ("physics",     CHECK_QUERY_MAP["physics"]),
    ("physics",     "yakit tuketimine gore CO2 emisyon faktoru fiziksel olarak tutarsiz"),
    ("peer",        CHECK_QUERY_MAP["peer"]),
    ("peer",        "benzer tesislere gore NOx SO2 emisyon yogunlugu cok dusuk emsal"),
    ("temporal",    CHECK_QUERY_MAP["temporal"]),
    ("temporal",    "surekli emisyon olcum zaman serisinde ani dusus anomali izleme"),
    ("measurement", "SEOS surekli olcum sistemi veri kaydi eksik beyan uyumsuzluk"),
    ("physics",     "stationary combustion IPCC 2006 emission factor tier 1 karbon"),
]

# --- INVALID: hicbir yasal kaynaga karsilik gelmemeli (alan-disi) ------------
INVALID = [
    "en iyi pizza tarifi nedir",
    "yarin istanbul hava durumu nasil olacak",
    "galatasaray fenerbahce derbi mac skoru",
    "telefon numarasi kac nasil ogrenirim",     # not: ham FAISS'te 0.584 almisti
    "en sevdigim sarki sozlerini yaz",
    "python ile web sitesi nasil yapilir",
    "bugun dolar euro kuru ne kadar",
    "kedi neden mirlar aciklama",
]

ALL_ROUTES = ["measurement", "physics", "peer", "temporal"]


def _top_score(index, query, check):
    pref = CHECK_GROUP_PREF.get(check)
    boost = {pref: _BOOST} if pref else None
    res = retrieve(index, query, k=1, sectors=None, boost_groups=boost)
    return res[0].score if res else 0.0


def main() -> None:
    print("Vektor indeksi hazirlaniyor (ilk calisma birkac dakika surebilir)...")
    index = get_embed_index(CORPUS)
    print(f"Indeks hazir: {len(index)} parca | model: {index.model_name}")
    print(f"Mevcut MIN_SCORE (recommend.py): {MIN_SCORE:.2f}\n")

    print("=" * 72)
    print("VALID sorgular (dogru kaynak beklenir) -- routing sonrasi en iyi skor")
    print("=" * 72)
    valid_scores = []
    for check, q in VALID:
        s = _top_score(index, q, check)
        valid_scores.append(s)
        print(f"  {s:5.3f}  [{check:11s}]  {q[:60]}")
    min_valid = min(valid_scores) if valid_scores else 0.0

    print("\n" + "=" * 72)
    print("INVALID sorgular (alan-disi) -- 4 rotanin en yuksegi (en kotu durum)")
    print("=" * 72)
    invalid_scores = []
    for q in INVALID:
        per_route = [(_top_score(index, q, r), r) for r in ALL_ROUTES]
        s, r = max(per_route, key=lambda x: x[0])
        invalid_scores.append(s)
        print(f"  {s:5.3f}  (rota: {r:11s})  {q[:52]}")
    max_invalid = max(invalid_scores) if invalid_scores else 0.0

    gap = min_valid - max_invalid
    print("\n" + "#" * 72)
    print(f"# MIN VALID   (en zayif dogru eslesme) : {min_valid:.3f}")
    print(f"# MAX INVALID (en guclu yanlis eslesme): {max_invalid:.3f}")
    print(f"# ARALIK (gap = MIN_VALID - MAX_INVALID): {gap:+.3f}")
    print("#" * 72)

    if gap > 0:
        rec = max_invalid + gap / 2.0            # temiz ayrimda: orta nokta
        print(f"\n[TEMIZ AYRIM] Guvenli aralik: ({max_invalid:.3f}, {min_valid:.3f}]")
        print(f"ONERILEN MIN_SCORE = {rec:.3f}   (orta nokta; istenirse "
              f"{max_invalid:.3f} uzerine biraz da konabilir)")
        print("-> Bu degeri recommend.py icindeki MIN_SCORE'a yazip "
              "run_rag_final_check.py calistirin.")
    else:
        print("\n[TEDBIR: CAKISMA/OVERLAP] MIN_VALID <= MAX_INVALID.")
        print("Tek bir esik VALID ve INVALID'i temiz ayirmiyor -> yapisal bulgu.")
        print(f"Muhafazakar secim MIN_VALID'e yakin (~{min_valid:.3f}); bazi dogru "
              "eslesmeler 'dayanak bulunamadi' alabilir. Korpus/routing gozden "
              "gecirilmeli; not olarak kaydedin.")


if __name__ == "__main__":
    main()
