# Proje kararları (docs/decisions)

- Veri: EPA CEMS — Limestone (Teksas), iki kömür birimi LIM1/LIM2, saatlik, 2025.
- CO2/HeatInput ≈ 0.105 sabit (CV~0.07) → karbon dengesi kontrolünün temeli.
- Mimari ayar-güdümlü + sütun-bağımsız → boyut esnekliği.
- Aylık toplulaştırma = ana kontrol referansı (beyan vs ölçüm).
