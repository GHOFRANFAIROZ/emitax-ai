"""
المرحلة ٨ — عرض عقد الإخراج للبلوكتشين.

    python run_export_demo.py

يشغّل النظام الكامل، يبني سجلّات التحقّق مع البصمات، يحفظ ملفّات التسليم
(onchain.jsonl / offchain.jsonl)، ويثبت مناعة البصمة ضدّ التعديل.
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
        print("شغّلي run_prepare.py أولاً."); return
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
    print(f"[تصدير] سجلّات: {len(onchain)}\n")

    print("[عيّنة on-chain payload] (يذهب للسلسلة — ملخّص + بصمة، بلا قيَم خام):")
    print(json.dumps(onchain[0], ensure_ascii=False, indent=2), "\n")

    # التحقّق من السلامة + مناعة ضدّ التعديل
    rec = offchain[0]
    print(f"[تحقّق] السجل الأصلي سليم؟ -> {verify_record(rec)}")
    tampered = dict(rec); tampered["declared"] = dict(tampered["declared"])
    first_gas = next(iter(tampered["declared"]))
    tampered["declared"][first_gas] *= 0.5          # تعديل خبيث لاحق
    print(f"[تحقّق] بعد تعديل قيمة مُصرَّحة، السجل سليم؟ -> {verify_record(tampered)} (لازم False)")

    # إعادة بناء نفس السجل تعطي نفس البصمة (حتمية)
    d0 = decisions.iloc[0].to_dict()
    decl0 = mixed[(mixed.facility == d0["facility"]) & (mixed.unit == d0["unit"]) & (mixed.ym == d0["ym"])].iloc[0].to_dict()
    meas0 = measurement[(measurement.facility == d0["facility"]) & (measurement.unit == d0["unit"]) & (measurement.ym == d0["ym"])].iloc[0].to_dict()
    again = build_record(d0, decl0, meas0, cfg, verified_at="2026-01-01T00:00:00+00:00")
    print(f"[حتمية] إعادة البناء تعطي نفس record_hash؟ -> {again['record_hash'] == rec['record_hash']}")

    outdir = Path(cfg.get("chain", {}).get("onchain_dir", cfg["data"]["processed_dir"]))
    (outdir / "onchain.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in onchain), encoding="utf-8")
    (outdir / "offchain.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in offchain), encoding="utf-8")
    print(f"\n[حفظ] {outdir/'onchain.jsonl'} (للبلوكتشين) , {outdir/'offchain.jsonl'} (خارج السلسلة)")
    print("المرحلة ٨ تمّت — النظام مكتمل من البيان حتى التسليم للبلوكتشين.")


if __name__ == "__main__":
    main()