"""Supheli beyani gercek resmi mevzuatla eslestirip Turkce, kaynaga dayali oneri uretir.

MIMARI:
- RAG kontrole bagli degildir; her ay her tesise calisir (bkz. recommend_all).
- Temiz kayit (kontrol yok) -> yasal UYGUNLUK teyidi (Turk mevzuatina referans).
- Supheli kayit -> tetiklenen HER kontrol icin ayri en iyi yasal kaynak getirilir
  (olcum->SKHKKY/SEOS, fizik->IPCC, emsal->EMEP, zaman->SEOS). Coklu kontrolde
  sorgu "corba"ya donmez; her kontrol kendi dogru kaynagina baglanir.
- Uydurma yok: her oneri gercek metni, belge basligini ve SAYFA numarasini alintilar.

GUVEN ESIGI (MIN_SCORE):
- Zayif eslesmeler "yasal dayanak" olarak YAYIMLANMAZ. Bir kontrol icin en iyi
  parcanin benzerligi MIN_SCORE altindaysa, o kaynak yerine
  "Yeterli guven duzeyinde mevzuat dayanagi bulunamadi." mesaji verilir.
- Esik, ROUTING SONRASI gorunen benzerlik skoru (RetrievedChunk.score, 0..1,
  kosinus) uzerinde uygulanir -- ham FAISS skoru degil; sektor/grup filtresi ve
  boost zaten uygulanmistir.
- MIN_SCORE degeri run_rag_calibration.py ciktisi (MIN VALID / MAX INVALID) ile
  kalibre edilir. Asagidaki deger BASLANGIC placeholder'idir.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .retrieve import RetrievedChunk, retrieve

# --- GUVEN ESIGI -------------------------------------------------------------
# Kalibrasyondan SONRA guncellenecek (run_rag_calibration.py -> onerilen esik).
# Gecici baslangic degeri; env "EMITAX_RAG_MIN_SCORE" ile de gecilebilir.
MIN_SCORE: float = 0.52
NO_BASIS_MSG: str = "Yeterli guven duzeyinde mevzuat dayanagi bulunamadi."


def _resolve_min_score(min_score: Optional[float]) -> float:
    if min_score is not None:
        return float(min_score)
    env = os.getenv("EMITAX_RAG_MIN_SCORE")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return MIN_SCORE
# -----------------------------------------------------------------------------

CHECK_QUERY_MAP: Dict[str, str] = {
    "measurement": "SKHKKY Ek-5 baca gazi emisyon sinir degeri asim mg/Nm3 beyan eksik beyan SEOS",
    "physics": "IPCC 2006 CO2 emisyon faktoru yakit tuketimi stationary combustion fiziksel tutarsiz",
    "peer": "EMEP emisyon faktoru enerji sektoru emsal karsilastirma NOx SO2 g/GJ",
    "temporal": "SEOS surekli emisyon olcum zaman serisi ani degisim izleme raporlama",
}
CHECK_QUERY_MAP.update({
    "measurement_check": CHECK_QUERY_MAP["measurement"], "physics_rules": CHECK_QUERY_MAP["physics"],
    "peer_iforest": CHECK_QUERY_MAP["peer"], "lstm_temporal": CHECK_QUERY_MAP["temporal"],
})

CHECK_LABELS: Dict[str, str] = {
    "measurement": "Olcum kontrolu (beyan vs SEOS / SKHKKY sinir)",
    "physics": "Fizik kurallari (IPCC 2006)",
    "peer": "Emsal kontrolu (EMEP)",
    "temporal": "Zamansal kontrol (SEOS)",
}
CHECK_LABELS.update({
    "measurement_check": CHECK_LABELS["measurement"], "physics_rules": CHECK_LABELS["physics"],
    "peer_iforest": CHECK_LABELS["peer"], "lstm_temporal": CHECK_LABELS["temporal"],
})

CHECK_GROUP_PREF: Dict[str, str] = {
    "measurement": "mevzuat", "physics": "ipcc", "peer": "emep", "temporal": "mevzuat",
    "measurement_check": "mevzuat", "physics_rules": "ipcc",
    "peer_iforest": "emep", "lstm_temporal": "mevzuat",
}
_BOOST = 1.6
_CLEAN_QUERY = "SKHKKY sanayi hava kirliligi kontrolu baca gazi emisyon sinir degeri uygunluk"


def _clean(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _split_checks(fired) -> List[str]:
    if fired is None:
        return []
    items = list(fired) if isinstance(fired, (list, tuple)) else re.split(r"[;,|/\s]+", str(fired))
    return [it.strip() for it in items if it and str(it).strip() and str(it).strip().lower() != "nan"]


def build_query(record: Dict) -> str:
    checks = _split_checks(record.get("fired_checks"))
    parts = [CHECK_QUERY_MAP.get(c, c) for c in checks]
    reason = _clean(record.get("reason"))
    if reason:
        parts.append(reason)
    if not parts:
        parts.append(_CLEAN_QUERY)
    return " ".join(parts)


@dataclass
class Recommendation:
    record: Dict
    query: str
    retrieved: List[RetrievedChunk]
    per_check: List[Tuple[str, RetrievedChunk]] = field(default_factory=list)
    weak_checks: List[Tuple[str, Optional[float]]] = field(default_factory=list)
    min_score: float = MIN_SCORE
    text: str = ""


def _header(record: Dict) -> str:
    return (f"Tesis: {_clean(record.get('facility')) or '?'} | "
            f"Unite: {_clean(record.get('unit')) or '?'} | "
            f"Donem: {_clean(record.get('ym')) or '?'}\n"
            f"Guven puani: {record.get('trust_score','?')} | "
            f"Karar: {_clean(record.get('action')) or '?'}")


def _quote(chunk) -> str:
    q = " ".join(chunk.text.split())
    return q[:297] + "..." if len(q) > 300 else q


def recommend(index, record: Dict, k: int = 2,
              sectors: Optional[Sequence[str]] = None,
              min_score: Optional[float] = None) -> Recommendation:
    ms = _resolve_min_score(min_score)
    checks = _split_checks(record.get("fired_checks"))
    reason = _clean(record.get("reason"))
    is_clean = len(checks) == 0
    per_check: List[Tuple[str, RetrievedChunk]] = []     # ESIGI GECEN kaynaklar
    weak_checks: List[Tuple[str, Optional[float]]] = []  # esik alti / eslesme yok
    flat: List[RetrievedChunk] = []
    seen = set()
    clean_best: Optional[float] = None

    if is_clean:
        got = retrieve(index, _CLEAN_QUERY, k=2, sectors=sectors,
                       boost_groups={"mevzuat": _BOOST}, only_groups=["mevzuat"])
        clean_best = got[0].score if got else None
        flat = [r for r in got if r.score >= ms]   # GUVEN ESIGI (routing sonrasi)
        query = _CLEAN_QUERY
    else:
        qparts = []
        for ch in checks:
            cq = CHECK_QUERY_MAP.get(ch, ch)
            if reason:
                cq = f"{cq} {reason}"
            qparts.append(cq)
            pref = CHECK_GROUP_PREF.get(ch)
            boost = {pref: _BOOST} if pref else None
            res = retrieve(index, cq, k=1, sectors=sectors, boost_groups=boost)
            if res and res[0].score >= ms:            # GUVEN ESIGI (routing sonrasi)
                r = res[0]
                per_check.append((ch, r))
                key = (r.chunk.doc_id, r.chunk.page)
                if key not in seen:
                    seen.add(key)
                    flat.append(r)
            else:
                weak_checks.append((ch, res[0].score if res else None))
        query = " | ".join(qparts)

    L: List[str] = []
    bar = "=" * 66
    baslik = "YASAL UYGUNLUK TEYIDI (RAG)" if is_clean else "YASAL DAYANAK ONERISI (RAG)"
    L += [bar, baslik, bar, _header(record)]
    L.append("Tetiklenen kontroller: " + (", ".join(CHECK_LABELS.get(c, c) for c in checks) or "-"))
    if reason:
        L.append(f"Bulgu (sistem): {reason}")
    L.append(f"Guven esigi (RAG): {ms:.2f}")
    L.append("")

    if is_clean:
        L.append("Ilgili yasal metin (resmi belgelerden alinti):")
        if flat:
            for i, r in enumerate(flat, 1):
                L.append(f"  [{i}] {r.chunk.title} -- sayfa {r.chunk.page}  [{r.chunk.group}]  (benzerlik: {r.score:.2f})")
                L.append(f"      \"{_quote(r.chunk)}\"")
                L.append(f"      Kaynak: {r.chunk.source}")
        else:
            L.append(f"  {NO_BASIS_MSG}")
            if clean_best is not None:
                L.append(f"  (en yakin eslesme benzerligi: {clean_best:.2f} < esik {ms:.2f})")
        L.append("")
        L.append("Oneri:")
        if flat:
            L.append(f"  Bu donemde tetiklenen kontrol yok; beyan ilgili mevzuatla uyumlu "
                     f"gorunmektedir. Referans: \"{flat[0].chunk.title}\" (sayfa {flat[0].chunk.page}). "
                     f"Rutin izlemeye devam edilmesi onerilir.")
        else:
            L.append("  Yeterli guvende referans bulunamadi; rutin izleme onerilir.")
    else:
        L.append("Her kontrol icin yasal dayanak (resmi belgelerden alinti):")
        if not per_check and not weak_checks:
            L.append("  (Eslesen yasal metin bulunamadi.)")
        for ch, r in per_check:
            L.append(f"  - {CHECK_LABELS.get(ch, ch)}:")
            L.append(f"      {r.chunk.title} -- sayfa {r.chunk.page}  [{r.chunk.group}]  (benzerlik: {r.score:.2f})")
            L.append(f"      \"{_quote(r.chunk)}\"")
            L.append(f"      Kaynak: {r.chunk.source}")
        for ch, best in weak_checks:
            L.append(f"  - {CHECK_LABELS.get(ch, ch)}:")
            L.append(f"      {NO_BASIS_MSG}")
            if best is not None:
                L.append(f"      (en yakin eslesme benzerligi: {best:.2f} < esik {ms:.2f})")
        L.append("")
        L.append("Oneri:")
        if per_check:
            L.append(f"  Tetiklenen kontroller icin yukaridaki resmi kaynaklar esas alinmalidir. "
                     f"Beyan, ilgili sinir deger/faktor ile karsilastirilmali; aykirilik "
                     f"dogrulanirsa tesisten resmi belge (SEOS kaydi, yakit faturasi, olcum "
                     f"raporu) talep edilmelidir.")
        else:
            L.append("  Yeterli yasal eslesme yok; insan incelemesi onerilir.")
    L.append(bar)

    return Recommendation(record=record, query=query, retrieved=flat,
                          per_check=per_check, weak_checks=weak_checks,
                          min_score=ms, text="\n".join(L))


def recommend_all(index, records, k: int = 2, sector_of=None,
                  min_score: Optional[float] = None) -> List[Recommendation]:
    out = []
    for rec in records:
        secs = sector_of(rec) if sector_of else None
        out.append(recommend(index, rec, k=k, sectors=secs, min_score=min_score))
    return out
