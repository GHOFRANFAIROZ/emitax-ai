"""
Aşama 8 — Blokzincir çıkış sözleşmesinin gösterimi.

    python run_export_demo.py

Tüm sistemi çalıştırır, parmak izleriyle doğrulama kayıtları oluşturur, teslim dosyalarını kaydeder
(onchain.jsonl / offchain.jsonl) ve parmak izinin değişikliğe karşı dayanıklılığını kanıtlar.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from emitax.config import load_config
from emitax.synth.declaration_gen import generate_honest, gas_list
from emitax.synth.tamper_gen import make_labeled_set
from emitax.verify.measurement_check import run_check, rollup_records
from emitax.verify.physics_rules import fit_params, check_physics, rollup_physics
from emitax.verify.temporal_check import check_temporal, rollup_temporal
from emitax.verify.peer_iforest import fit_iforest, check_peer_iforest, rollup_peer_iforest
from emitax.verify.fusion import decide
from emitax.chain.record import export_decisions, verify_record, build_record


def main() -> None:
    cfg = load_config("configs/default.yaml")
    mpath = Path(cfg["data"]["processed_dir"]) / "monthly_measurement.csv"
    if not mpath.exists():
        print("önce run_prepare.py çalıştırın."); return
    measurement = pd.read_csv(mpath)
    gases = gas_list(cfg, measurement)

    honest = generate_honest(measurement, cfg)
    mixed = make_labeled_set(honest, cfg, gases)
    rollups = {
        "measurement": rollup_records(run_check(mixed, measurement, cfg)),
        "physics":     rollup_physics(check_physics(mixed, fit_params(measurement, cfg), cfg)),
        "temporal":    rollup_temporal(check_temporal(mixed, cfg)),
        "peer":        rollup_peer_iforest(check_peer_iforest(mixed, fit_iforest(measurement, cfg), cfg)),
    }
    decisions = decide(rollups, mixed, cfg)

    onchain, offchain = export_decisions(decisions, mixed, measurement, cfg,
                                         verified_at="2026-01-01T00:00:00+00:00")
    print(f"[Dışa aktarım] kayıt: {len(onchain)}\n")

    print("[Örnek on-chain payload] (zincire gider — özet + parmak izi, ham değer yok):")
    print(json.dumps(onchain[0], ensure_ascii=False, indent=2), "\n")

    # Bütünlük doğrulaması + değişikliğe karşı dayanıklılık
    rec = offchain[0]
    print(f"[Doğrulama] orijinal kayıt sağlam mı? -> {verify_record(rec)}")
    tampered = dict(rec); tampered["declared"] = dict(tampered["declared"])
    first_gas = next(iter(tampered["declared"]))
    tampered["declared"][first_gas] *= 0.5          # sonradan yapılan kötücül değişiklik
    print(f"[Doğrulama] beyan değeri değiştirildikten sonra kayıt sağlam mı? -> {verify_record(tampered)} (False olmalı)")

    # Aynı kaydı yeniden inşa etmek aynı parmak izini verir (belirlenimci)
    d0 = decisions.iloc[0].to_dict()
    decl0 = mixed[(mixed.facility == d0["facility"]) & (mixed.unit == d0["unit"]) & (mixed.ym == d0["ym"])].iloc[0].to_dict()
    meas0 = measurement[(measurement.facility == d0["facility"]) & (measurement.unit == d0["unit"]) & (measurement.ym == d0["ym"])].iloc[0].to_dict()
    again = build_record(d0, decl0, meas0, cfg, verified_at="2026-01-01T00:00:00+00:00")
    print(f"[Belirlenimcilik] yeniden inşa aynı record_hash'i veriyor mu? -> {again['record_hash'] == rec['record_hash']}")

    outdir = Path(cfg.get("chain", {}).get("onchain_dir", cfg["data"]["processed_dir"]))
    (outdir / "onchain.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in onchain), encoding="utf-8")
    (outdir / "offchain.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in offchain), encoding="utf-8")
    print(f"\n[Kaydedildi] {outdir/'onchain.jsonl'} (blokzincir için) , {outdir/'offchain.jsonl'} (zincir dışı)")
    print("Aşama 8 tamamlandı — sistem beyandan blokzincire teslime kadar eksiksiz.")


if __name__ == "__main__":
    main()