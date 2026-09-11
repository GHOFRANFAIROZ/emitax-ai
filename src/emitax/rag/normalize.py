"""Metin normalizasyon yardimcisi (RAG icin).

Alt simge rakamlarini (ornek: SO2 icindeki alt simge 2) normal rakama cevirir
ve metni kucuk harfe indirger. Boylece 'SO2' yazilisinin farkli bicimleri
ayni sekilde eslesir. Bu fonksiyon hem indeksleme hem sorgu tarafinda ayni
sekilde uygulanir (TfidfVectorizer preprocessor olarak verilir).
"""

# Unicode alt simge rakamlari -> normal rakamlar
_SUBSCRIPTS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089",
                            "0123456789")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).translate(_SUBSCRIPTS).lower()
