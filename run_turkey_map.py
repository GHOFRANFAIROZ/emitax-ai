"""
Türkiye Emisyon Haritası — Climate TRACE gerçek verisiyle (2025).

Girdi:  data/turkey_climatetrace.csv   (Climate TRACE asset/year, sektör: Power)
Çıktı:  docs/analysis/turkey_emisyon_haritasi.html   (tarayıcıda: yakınlaştır/kaydır)

Kabarcık boyutu = yıllık CO₂e emisyonu (gerçek). Renk = yakıt türü.
Not: emissions_factor sütunu = t CO₂e / MWh = bizim "yoğunluk" ölçümüzün aynısı
(akran kontrolünün karşılaştırdığı değer).

Veri kaynağı: Climate TRACE (climatetrace.org), CC-BY 4.0.
"""
from __future__ import annotations
from pathlib import Path
import math
import pandas as pd

try:
    import folium
except ImportError:
    raise SystemExit("folium gerekli. Kurulum: pip install folium")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "analysis" / "turkey_emisyon_haritasi.html"


def find_csv() -> Path | None:
    d = ROOT / "data"
    cand = d / "turkey_climatetrace.csv"
    if cand.exists():
        return cand
    for p in sorted(d.glob("*emissions*co2e*.csv")):
        return p
    return None


def fuel_category(source_type: str) -> tuple[str, str]:
    """Yakıt türünü sadeleştir → (etiket, renk). Kömür varsa kömür sayılır."""
    s = str(source_type).lower()
    if "coal" in s:
        return "Kömür", "#c62828"
    if "biomass" in s:
        return "Biyokütle", "#2e7d32"
    if "oil" in s and "gas" not in s:
        return "Petrol", "#6a1b9a"
    if "gas" in s or "fossil" in s:
        return "Gaz / diğer fosil", "#1565c0"
    return "Diğer", "#757575"


def radius(emissions_t: float) -> float:
    """Yarıçap = emisyonla orantılı (kök ile yumuşatılmış). Sıfır → küçük nokta."""
    if not emissions_t or emissions_t <= 0:
        return 4.0
    return max(5.0, math.sqrt(emissions_t / 1e6) * 9.0)  # Mt ölçeğinde


def main():
    csv = find_csv()
    if not csv:
        print("[Hata] data/turkey_climatetrace.csv bulunamadı.")
        return
    df = pd.read_csv(csv)
    df = df.dropna(subset=["lat", "lon"])

    m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"),
        attr="Esri", name="Esri Light Gray", control=False,
    ).add_to(m)

    for _, r in df.iterrows():
        label, color = fuel_category(r.get("source_type", ""))
        emis = float(r.get("emissions_quantity") or 0)
        mt = emis / 1e6
        ef = r.get("emissions_factor")
        ef_txt = f"{float(ef):.3f} t CO₂e/MWh" if pd.notna(ef) else "veri yok"
        cap = r.get("capacity")
        cap_txt = f"{float(cap):,.0f} MW" if pd.notna(cap) else "veri yok"
        popup = folium.Popup(html=(
            f"<b>{r.get('source_name','?')}</b><br>"
            f"Yakıt: {label}<br>"
            f"Yıllık CO₂e: <b>{mt:,.2f} Mt</b><br>"
            f"Kapasite: {cap_txt}<br>"
            f"Yoğunluk (emisyon/enerji): {ef_txt}<br>"
            f"<i>Kaynak: Climate TRACE 2025</i>"), max_width=280)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=radius(emis),
            color=color, fill=True, fill_color=color, fill_opacity=0.6, weight=1,
            popup=popup, tooltip=f"{r.get('source_name','?')} — {mt:,.2f} Mt",
        ).add_to(m)

    title_html = """
    <div style="position:fixed; top:10px; left:50px; z-index:9999; background:white;
         padding:8px 14px; border:1px solid #888; border-radius:6px; font-family:sans-serif;">
      <b>Türkiye — Elektrik Santrali Emisyonları (2025)</b><br>
      <span style="font-size:12px;">Kabarcık boyutu = yıllık CO₂e</span><br>
      <span style="color:#c62828;">●</span> Kömür &nbsp;
      <span style="color:#1565c0;">●</span> Gaz &nbsp;
      <span style="color:#2e7d32;">●</span> Biyokütle &nbsp;
      <span style="color:#6a1b9a;">●</span> Petrol<br>
      <span style="font-size:11px;color:#555;">Kaynak: Climate TRACE (CC-BY 4.0)</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT))

    emitters = df[df["emissions_quantity"] > 0]
    print(f"[Kaydedildi] {OUT}")
    print(f"   Toplam {len(df)} santral haritalandı ({len(emitters)} emisyon > 0).")
    print(f"   Toplam emisyon: {df['emissions_quantity'].sum()/1e6:,.1f} Mt CO₂e (2025)")
    print("\n   En yüksek 5 kaynak:")
    for _, r in emitters.nlargest(5, "emissions_quantity").iterrows():
        print(f"     {r['source_name']:<38} {r['emissions_quantity']/1e6:>6.2f} Mt")
    print("\n   Tarayıcıda aç:  start docs\\analysis\\turkey_emisyon_haritasi.html")


if __name__ == "__main__":
    main()
