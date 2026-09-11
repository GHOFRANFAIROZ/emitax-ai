"""Vesika parcalama (gercek resmi PDF'lerden cikarilan metin + manifest).

legal_corpus/ altindaki .txt dosyalarini okur. Her dosya [[SAYFA n]]
isaretleriyle sayfalara ayrilmistir. manifest.json her belgenin basligini,
grubunu (mevzuat/met/emep/ipcc/eu_brite), sektor bagimsiz olup olmadigini ve
sektor etiketlerini tasir. Parca (chunk) sayfa numarasi + kaynak bilgisi tasir.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_MAX_CHARS = 1100

# Grup -> okunabilir kaynak etiketi (atif icin)
GROUP_SOURCE = {
    "mevzuat": "Turk resmi mevzuati (mevzuat.gov.tr / Resmi Gazete)",
    "met": "MET Tebligi -- Mevcut En Iyi Teknikler (Cevre Bakanligi)",
    "emep": "EMEP/EEA 2023 Rehberi (uluslararasi)",
    "ipcc": "IPCC 2006 Kilavuzu (uluslararasi)",
    "eu_brite": "EU-BRITE ROM emisyon izleme yonergesi (AB)",
    "diger": "Yasal belge",
}


@dataclass
class Chunk:
    doc_id: str
    title: str
    group: str
    page: int
    text: str
    source: str
    sector_independent: bool = True
    sectors: List[str] = field(default_factory=list)

    def as_context(self) -> str:
        return f"{self.title}\n{self.text}".strip()


def _split_long(text: str, max_chars: int = _MAX_CHARS) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts, buf, size = [], [], 0
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if size + len(para) > max_chars and buf:
            parts.append("\n\n".join(buf)); buf, size = [], 0
        buf.append(para); size += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


_PAGE_RE = re.compile(r"\[\[SAYFA\s+(\d+)\]\]")


def _iter_pages(raw: str):
    # metni [[SAYFA n]] isaretlerine gore sayfalara ayirir
    positions = [(m.start(), int(m.group(1)), m.end()) for m in _PAGE_RE.finditer(raw)]
    if not positions:
        yield 1, raw
        return
    for i, (start, page, end) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
        yield page, raw[end:stop]


def load_corpus(corpus_dir) -> List[Chunk]:
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json bulunamadi: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: List[Chunk] = []
    for entry in manifest:
        fp = corpus_dir / entry["file"]
        if not fp.exists():
            continue
        raw = fp.read_text(encoding="utf-8")
        src = GROUP_SOURCE.get(entry["group"], "Yasal belge")
        for page, page_text in _iter_pages(raw):
            for piece in _split_long(page_text):
                if len(piece.strip()) < 40:
                    continue
                chunks.append(Chunk(
                    doc_id=entry["file"], title=entry["title"], group=entry["group"],
                    page=page, text=piece, source=src,
                    sector_independent=entry.get("sector_independent", True),
                    sectors=entry.get("sectors", []),
                ))
    return chunks
