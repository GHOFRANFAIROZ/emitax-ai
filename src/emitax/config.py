from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """قراءة الإعدادات من YAML. دالة I/O واحدة."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
