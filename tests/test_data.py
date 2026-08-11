from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from emitax.data import loader, aggregate

CFG = {
    "columns": {
        "facility": "Facility Name", "facility_id": "Facility ID", "unit": "Unit ID",
        "date": "Date", "hour": "Hour", "operating_time": "Operating Time",
        "fuel_type": "Primary Fuel Type", "drop_all_nan": True,
        "context": {"heat_input": "Heat Input (mmBtu)", "gross_load": "Gross Load (MW)"},
        "gases": {
            "CO2": {"mass": "CO2 Mass (short tons)", "rate": "CO2 Rate (short tons/mmBtu)"},
            "SO2": {"mass": "SO2 Mass (lbs)", "rate": "SO2 Rate (lbs/mmBtu)"},
            "NOx": {"mass": "NOx Mass (lbs)", "rate": "NOx Rate (lbs/mmBtu)"},
        },
    },
    "aggregate": {"period": "M"},
}


def _raw():
    return pd.DataFrame({
        " Facility Name ": ["A", "A", "A"],          # فراغات مقصودة لاختبار strip
        "Facility ID": [1, 1, 1],
        "Unit ID": ["U1", "U1", "U1"],
        "Date": ["2025-01-01", "2025-01-01", "2025-02-01"],
        "Hour": [0, 1, 0],
        "Operating Time": [1, 1, 1],
        "Heat Input (mmBtu)": [100.0, 200.0, 50.0],
        "Gross Load (MW)": [10, 20, 5],
        "CO2 Mass (short tons)": [10.0, 20.0, 5.0],
        "SO2 Mass (lbs)": [1.0, 2.0, 0.5],
        "NOx Mass (lbs)": [3.0, 4.0, 1.0],
        "Primary Fuel Type": ["Coal", "Coal", "Coal"],
        "Steam Load (1000 lb/hr)": [None, None, None],   # فارغ كلياً
        "Some Extra Column": ["x", "y", "z"],            # عمود زائد يُتجاهَل
    })


def test_strip_and_select_ignores_extra_and_missing():
    raw = loader.strip_cols(_raw())
    tidy, missing = loader.select_and_rename(raw, CFG)
    assert "facility" in tidy.columns and "CO2_mass" in tidy.columns
    assert "Some Extra Column" not in tidy.columns          # الزائد يُتجاهَل
    assert isinstance(missing, list)


def test_timestamp_and_drop_all_nan():
    raw = loader.strip_cols(_raw())
    tidy, _ = loader.select_and_rename(raw, CFG)
    tidy = loader.coerce_numeric(tidy, CFG)
    tidy = loader.build_timestamp(tidy)
    tidy = tidy.dropna(axis=1, how="all")
    assert "ts" in tidy.columns
    assert str(tidy["ts"].iloc[1]) == "2025-01-01 01:00:00"


def test_monthly_aggregate_sums_correctly():
    raw = loader.strip_cols(_raw())
    tidy, _ = loader.select_and_rename(raw, CFG)
    tidy = loader.coerce_numeric(tidy, CFG)
    tidy = loader.build_timestamp(tidy)
    m = aggregate.aggregate_monthly(tidy, CFG)
    jan = m[m["ym"] == "2025-01"].iloc[0]
    assert jan["CO2_mass"] == pytest.approx(30.0)     # 10 + 20
    assert jan["heat_input"] == pytest.approx(300.0)  # 100 + 200
    feb = m[m["ym"] == "2025-02"].iloc[0]
    assert feb["CO2_mass"] == pytest.approx(5.0)


def test_measurement_availability_flags_present():
    raw = loader.strip_cols(_raw())
    tidy, _ = loader.select_and_rename(raw, CFG)
    tidy = loader.coerce_numeric(tidy, CFG)
    tidy = loader.build_timestamp(tidy)
    av = aggregate.measurement_available(tidy, CFG)
    assert bool(av["cems_available"].iloc[0]) is True
