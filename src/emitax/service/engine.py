"""
Emitax verification engine (service layer, Phase A).

Self-contained: it loads the artifacts produced by run_prepare.py and the
Colab export (monthly_measurement.csv, hourly_clean.parquet, lstm_ae.onnx,
lstm_meta.json) plus configs/default.yaml, and re-applies the SAME four
checks + fusion described in the report (§5.1/§5.2/§3.3) to a single
declaration. It depends only on artifacts + config, not on the internal
signatures of the batch modules — so it runs even if those evolve.

Checks
  1. measurement  : declared vs aggregated CEMS measurement (rel gap < -tol)
  2. physics      : IPCC 2006 CO2 factor floor + gas-ratio low bounds
  3. peer         : Isolation Forest on sector intensity (directional-low)
  4. temporal     : LSTM-AE reconstruction error per (unit, month) via ONNX

Fusion: trust = 100 - Σ weights(fired) ; 3-tier decision from config.
"""
from __future__ import annotations

import json
import hashlib
import logging
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("emitax.service.engine")

_GAS_KEYS = ["co2", "so2", "nox"]  # PM is not measured by CEMS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, *keywords: str, prefer: Optional[str] = None) -> Optional[str]:
    """Fuzzy-locate a column by keyword(s); optionally prefer names containing `prefer`."""
    kws = [k.lower() for k in keywords]
    cands = [c for c in df.columns if all(k in str(c).lower() for k in kws)]
    if not cands:
        return None
    if prefer:
        pref = [c for c in cands if prefer.lower() in str(c).lower()]
        if pref:
            return pref[0]
    # prefer mass over rate by default for emission columns
    mass = [c for c in cands if "mass" in str(c).lower()]
    return (mass or cands)[0]


def _to_ym(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _cfg_get(cfg: dict, path: List[str], default):
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class VerificationEngine:
    def __init__(self, cfg: dict, paths: Dict[str, str]):
        self.cfg = cfg
        self.paths = paths

        # tunables (fall back to the values used across the phases)
        self.meas_tol = float(_cfg_get(cfg, ["check", "rel_tolerance"], 0.05))
        self.phys_tol = float(_cfg_get(cfg, ["physics", "rel_tolerance"], 0.10))
        self.ratio_low_q = float(_cfg_get(cfg, ["physics", "ratio_low_quantile"], 0.05))
        self.peer_low_q = float(_cfg_get(cfg, ["peer", "low_quantile"], 0.05))
        self.peer_min = int(_cfg_get(cfg, ["peer_iforest", "min_peers"],
                                     _cfg_get(cfg, ["peer", "min_peers"], 2)))
        self.ipcc = _cfg_get(cfg, ["physics_ipcc", "co2_factor_by_fuel"], {}) or {}
        self.ipcc_default = float(_cfg_get(cfg, ["physics_ipcc", "default_factor"], 0.1049))

        self.weights = _cfg_get(cfg, ["fusion", "weights"],
                                {"measurement": 45, "physics": 30, "peer": 25, "temporal": 20})
        self.high_thr = float(_cfg_get(cfg, ["fusion", "high_threshold"], 85))
        self.med_thr = float(_cfg_get(cfg, ["fusion", "medium_threshold"], 55))
        self.schema_version = _cfg_get(cfg, ["chain", "schema_version"], "emitax-ai/1.0")

        # loaded state
        self.measurement: Optional[pd.DataFrame] = None
        self.meas_cols: Dict[str, str] = {}
        self.meas_index: Dict[Tuple[str, str, str], pd.Series] = {}
        self.physics_factor: float = self.ipcc_default
        self.ratio_bounds: Dict[str, float] = {}
        self.iforest = None
        self.peer_medians: Dict[str, float] = {}
        self.temporal_flags: Dict[Tuple[str, str], bool] = {}
        self.checks_available: Dict[str, bool] = {
            "measurement": False, "physics": False, "peer": False, "temporal": False}

        self._load_measurement()
        self._fit_physics()
        self._fit_peer()
        self._precompute_temporal()

    # ---------------- load reference measurement ----------------
    def _load_measurement(self):
        p = Path(self.paths["measurement_path"])
        if not p.exists():
            log.warning("measurement reference not found: %s", p)
            return
        df = pd.read_csv(p)
        self.measurement = df

        fac = _find_col(df, "facility") or _find_col(df, "plant")
        unit = _find_col(df, "unit")
        # id columns fall back to constants if absent
        self.meas_cols["facility"] = fac
        self.meas_cols["unit"] = unit
        for g in _GAS_KEYS:
            self.meas_cols[g] = _find_col(df, g, prefer="mass")
        self.meas_cols["heat"] = _find_col(df, "heat")
        # period key
        ym = _find_col(df, "ym") or _find_col(df, "period")
        ycol, mcol = _find_col(df, "year"), _find_col(df, "month")
        self.meas_cols["ym"], self.meas_cols["year"], self.meas_cols["month"] = ym, ycol, mcol

        def _row_ym(r) -> str:
            if ym:
                return str(r[ym])[:7]
            if ycol and mcol:
                return _to_ym(r[ycol], r[mcol])
            return ""

        for _, r in df.iterrows():
            fid = str(r[fac]) if fac else "FAC"
            uid = str(r[unit]) if unit else "UNIT"
            self.meas_index[(fid, uid, _row_ym(r))] = r
        self.checks_available["measurement"] = bool(self.meas_cols.get("co2"))
        log.info("measurement reference: %d rows, cols=%s", len(df), self.meas_cols)

    # ---------------- physics (IPCC factor + ratio bounds) ----------------
    def _fit_physics(self):
        df = self.measurement
        # learn low ratio bounds (so2/co2, nox/co2) from real measurement if present
        if df is not None and self.meas_cols.get("co2"):
            co2 = pd.to_numeric(df[self.meas_cols["co2"]], errors="coerce")
            for g in ("so2", "nox"):
                gc = self.meas_cols.get(g)
                if gc:
                    ratio = pd.to_numeric(df[gc], errors="coerce") / co2.replace(0, np.nan)
                    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(ratio) >= 3:
                        self.ratio_bounds[g] = float(ratio.quantile(self.ratio_low_q))
        self.checks_available["physics"] = True  # IPCC factor is always available
        log.info("physics ready: factor(default)=%.4f ratio_bounds=%s",
                 self.ipcc_default, self.ratio_bounds)

    def _factor_for(self, fuel: str) -> float:
        if not fuel:
            return self.ipcc_default
        f = str(fuel).lower()
        for k, v in self.ipcc.items():
            if str(k).lower() in f or f in str(k).lower():
                return float(v)
        return self.ipcc_default

    # ---------------- peer (Isolation Forest, directional-low) ----------------
    def _fit_peer(self):
        df = self.measurement
        if df is None or not self.meas_cols.get("heat"):
            return
        hi = pd.to_numeric(df[self.meas_cols["heat"]], errors="coerce")
        feats, rows = [], None
        cols = []
        for g in _GAS_KEYS:
            gc = self.meas_cols.get(g)
            if gc:
                inten = pd.to_numeric(df[gc], errors="coerce") / hi.replace(0, np.nan)
                feats.append(inten.rename(g))
                cols.append(g)
        if not feats:
            return
        X = pd.concat(feats, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(X) < self.peer_min:
            log.warning("peer: not enough rows (%d) to fit Isolation Forest", len(X))
            return
        try:
            from sklearn.ensemble import IsolationForest
            self.iforest = IsolationForest(
                n_estimators=int(_cfg_get(self.cfg, ["peer_iforest", "n_estimators"], 200)),
                contamination=_cfg_get(self.cfg, ["peer_iforest", "contamination"], "auto"),
                random_state=42,
            ).fit(X.values)
            self.peer_cols = cols
            self.peer_medians = {g: float(X[g].median()) for g in cols}
            self.checks_available["peer"] = True
            log.info("peer Isolation Forest fitted on %d rows, gases=%s", len(X), cols)
        except Exception as e:  # pragma: no cover
            log.warning("peer fit failed: %s", e)

    # ---------------- temporal (LSTM-AE recon error via ONNX) ----------------
    def _precompute_temporal(self):
        onnx_p, meta_p = Path(self.paths["onnx_path"]), Path(self.paths["meta_path"])
        hourly_p = Path(self.paths["hourly_path"])
        if not (onnx_p.exists() and meta_p.exists() and hourly_p.exists()):
            log.warning("temporal disabled (missing onnx/meta/hourly)")
            return
        try:
            import onnxruntime as ort
            meta = json.loads(meta_p.read_text())
            feats = meta["feats"]
            mean = np.asarray(meta["mean"], dtype=np.float32)
            std = np.asarray(meta["std"], dtype=np.float32)
            thr = float(meta["threshold"])
            seq = int(meta["seq_len"])

            hourly = pd.read_parquet(hourly_p)
            ucol = _find_col(hourly, "unit") or "unit_id"
            fcol = _find_col(hourly, "facility")
            tcol = (_find_col(hourly, "date") or _find_col(hourly, "time")
                    or _find_col(hourly, "timestamp"))
            missing = [c for c in feats if c not in hourly.columns]
            if missing:
                log.warning("temporal disabled (feats missing in hourly: %s)", missing)
                return

            sess = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name

            for uid, g in hourly.groupby(ucol):
                gg = g.dropna(subset=feats)
                if tcol:
                    gg = gg.sort_values(tcol)
                if len(gg) < seq:
                    continue
                arr = ((gg[feats].to_numpy(dtype=np.float32) - mean) / std)
                ts = pd.to_datetime(gg[tcol]) if tcol else None
                n = (len(arr) // seq) * seq
                if n == 0:
                    continue
                wins = arr[:n].reshape(-1, seq, len(feats))          # [W, seq, F]
                recon = sess.run(None, {in_name: wins})[0]
                err = ((recon - wins) ** 2).mean(axis=(1, 2))          # per-window
                # month of each window = month of its last row
                if ts is not None:
                    last_idx = (np.arange(len(wins)) + 1) * seq - 1
                    yms = ts.iloc[last_idx].dt.strftime("%Y-%m").to_numpy()
                else:
                    yms = np.array([""] * len(wins))
                fid = str(g[fcol].iloc[0]) if fcol else "FAC"
                per = pd.DataFrame({"ym": yms, "err": err})
                agg = per.groupby("ym")["err"].quantile(0.95)
                for ym, e in agg.items():
                    self.temporal_flags[(str(uid), str(ym))] = bool(e > thr)
            self.checks_available["temporal"] = True
            log.info("temporal ready: %d (unit,month) recon flags", len(self.temporal_flags))
        except Exception as e:  # pragma: no cover
            log.warning("temporal precompute failed: %s", e)

    # ---------------- per-declaration checks ----------------
    def _check_measurement(self, d) -> Tuple[Optional[bool], str]:
        key = (d.facility_id, d.unit_id, _to_ym(d.year, d.month))
        row = self.meas_index.get(key)
        if row is None or not self.checks_available["measurement"]:
            return None, "ölçüm yok (CEMS bulunmuyor)"
        worst = None
        for g in _GAS_KEYS:
            gc = self.meas_cols.get(g)
            dv = getattr(d, g)
            if gc is None or dv is None:
                continue
            mv = pd.to_numeric(pd.Series([row[gc]]), errors="coerce").iloc[0]
            if pd.isna(mv) or mv == 0:
                continue
            rel = (float(dv) - float(mv)) / float(mv)
            if rel < -self.meas_tol:
                worst = min(worst, rel) if worst is not None else rel
        if worst is not None:
            return True, f"beyan ölçümün altında (en büyük açık %{abs(worst)*100:.0f})"
        return False, "beyan ölçümle uyumlu"

    def _check_physics(self, d) -> Tuple[bool, str]:
        factor = self._factor_for(d.fuel_type)
        floor = factor * float(d.heat_input)
        if float(d.co2) < floor * (1 - self.phys_tol):
            return True, f"CO2 fiziksel taban altında (asgari≈{floor:,.0f})"
        for g in ("so2", "nox"):
            bnd = self.ratio_bounds.get(g)
            dv = getattr(d, g)
            if bnd and dv is not None and d.co2:
                if (float(dv) / float(d.co2)) < bnd * (1 - self.phys_tol):
                    return True, f"{g.upper()}/CO2 oranı beklenenin altında"
        return False, "fiziksel olarak tutarlı"

    def _check_peer(self, d) -> Tuple[Optional[bool], str]:
        if self.iforest is None or not float(d.heat_input):
            return None, "akran havuzu yok"
        vec, low = [], False
        for g in getattr(self, "peer_cols", []):
            dv = getattr(d, g)
            inten = (float(dv) / float(d.heat_input)) if dv is not None else np.nan
            vec.append(inten)
            if not np.isnan(inten) and inten < self.peer_medians.get(g, np.inf):
                low = True
        vec = np.asarray([vec], dtype=float)
        if np.isnan(vec).any():
            return None, "akran karşılaştırması için eksik veri"
        outlier = self.iforest.predict(vec)[0] == -1
        flagged = bool(outlier and low)   # directional: only under-reporting
        return flagged, ("sektör dağılımından anormal düşük" if flagged
                         else "akranlarla uyumlu")

    def _check_temporal(self, d) -> Tuple[Optional[bool], str]:
        if not self.checks_available["temporal"]:
            return None, "zamansal model yok"
        f = self.temporal_flags.get((d.unit_id, _to_ym(d.year, d.month)))
        if f is None:
            return None, "bu dönem için geçmiş penceresi yok"
        return bool(f), ("olağan dışı dönem (yüksek yeniden kurma hatası)"
                         if f else "geçmiş örüntüyle tutarlı")

    # ---------------- fusion + record ----------------
    def _fuse(self, flags: Dict[str, Optional[bool]]) -> Tuple[float, str, List[str]]:
        fired = [k for k, v in flags.items() if v is True]
        penalty = sum(float(self.weights.get(k, 0)) for k in fired)
        trust = max(0.0, 100.0 - penalty)
        if trust >= self.high_thr:
            decision = "approve"
        elif trust >= self.med_thr:
            decision = "request_docs"
        else:
            decision = "human_review"
        return trust, decision, fired

    def _reason_tr(self, fired: List[str]) -> str:
        lbl = {"measurement": "Beyan ile ölçüm uyuşmuyor",
               "physics": "Fiziksel tutarsızlık",
               "peer": "Akranlardan sapma",
               "temporal": "Zamansal anomali"}
        if not fired:
            return "Beyan tüm kontrolleri geçti."
        return "Güven skoru şu nedenlerle düştü: " + ", ".join(lbl.get(k, k) for k in fired) + "."

    def _record(self, d, trust, decision, fired) -> Tuple[str, Dict[str, Any]]:
        declared = {g: getattr(d, g) for g in _GAS_KEYS + ["pm"]}
        declared["heat_input"] = d.heat_input
        data_hash = hashlib.sha256(
            json.dumps(declared, sort_keys=True, default=str).encode()).hexdigest()
        payload = {
            "schema_version": self.schema_version,
            "facility_id": d.facility_id, "unit_id": d.unit_id,
            "ym": _to_ym(d.year, d.month),
            "trust_score": round(trust, 2), "decision": decision,
            "fired_checks": fired, "data_hash": data_hash,
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        record_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        payload["record_hash"] = record_hash
        return record_hash, payload

    # ---------------- public API ----------------
    def verify(self, d) -> Dict[str, Any]:
        m_flag, m_txt = self._check_measurement(d)
        p_flag, p_txt = self._check_physics(d)
        r_flag, r_txt = self._check_peer(d)
        t_flag, t_txt = self._check_temporal(d)

        flags = {"measurement": m_flag, "physics": p_flag,
                 "peer": r_flag, "temporal": t_flag}
        trust, decision, fired = self._fuse(flags)
        record_hash, payload = self._record(d, trust, decision, fired)

        return {
            "facility_id": d.facility_id, "unit_id": d.unit_id,
            "ym": _to_ym(d.year, d.month),
            "trust_score": round(trust, 2), "decision": decision,
            "fired_checks": fired, "reason": self._reason_tr(fired),
            "per_check": {
                "measurement": {"flagged": m_flag, "detail": m_txt},
                "physics": {"flagged": p_flag, "detail": p_txt},
                "peer": {"flagged": r_flag, "detail": r_txt},
                "temporal": {"flagged": t_flag, "detail": t_txt},
            },
            "record_hash": record_hash,
            "onchain_payload": payload,
        }


# --------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------
# .../emitax-ai/src/emitax/service/engine.py -> parents[3] = emitax-ai
_AI_ROOT = Path(__file__).resolve().parents[3]


def build_engine(config_path: str | Path | None = None) -> VerificationEngine:
    """Load config + artifact paths and construct the engine (called at startup).

    Both the config file and the artifact paths are resolved against the
    emitax-ai package root, not the process working directory. The service is a
    standalone unit: it must load the same artifacts no matter where it is
    started from.
    """
    from emitax.config import load_config
    cfg = load_config(config_path or (_AI_ROOT / "configs" / "default.yaml"))
    svc = _cfg_get(cfg, ["service"], {}) or {}
    defaults = {
        "measurement_path": "data/processed/monthly_measurement.csv",
        "hourly_path": "data/processed/hourly_clean.parquet",
        "onnx_path": "data/processed/lstm_ae.onnx",
        "meta_path": "data/processed/lstm_meta.json",
    }
    paths = {}
    for key, default in defaults.items():
        value = Path(svc.get(key, default))
        paths[key] = str(value if value.is_absolute() else (_AI_ROOT / value))
    return VerificationEngine(cfg, paths)
