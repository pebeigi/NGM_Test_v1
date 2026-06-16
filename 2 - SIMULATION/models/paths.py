from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Project root = parent of the `models/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

MODELS_DIR: Path = PROJECT_ROOT / "models"
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

MODEL_PARAMS_DIR: Path = MODELS_DIR / "model_params"

MOBIL_RESULTS_PATH: Path = MODEL_PARAMS_DIR / "MOBIL_results.csv"

# Simulation vector: [Disc p, Disc a_th, Disc b_safe, Mand b_safe]
MOBIL_VECTOR_COLUMNS: Sequence[str] = (
    "Discretionary_p_optimal",
    "Discretionary_a_th",
    "Discretionary_b_safe",
    "Mandatory_b_safe",
)

# GUI LC_Parameters keys -> canonical MOBIL_results.csv column name.
MOBIL_GUI_TO_CSV: Dict[str, str] = {
    "Disc: p_opt": "Discretionary_p_optimal",
    "Disc: a_th": "Discretionary_a_th",
    "Disc: b_safe": "Discretionary_b_safe",
    "Mand: b_safe": "Mandatory_b_safe",
}

# Legacy exports used spaced headers.
MOBIL_CSV_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "Discretionary_p_optimal": ("Discretionary_p_optimal", "Discretionary_p optimal"),
    "Discretionary_a_th": ("Discretionary_a_th",),
    "Discretionary_b_safe": ("Discretionary_b_safe",),
    "Mandatory_b_safe": ("Mandatory_b_safe",),
    "Mandatory_p_optimal": ("Mandatory_p_optimal", "Mandatory_p optimal"),
    "Mandatory_a_th": ("Mandatory_a_th",),
}

MOBIL_VECTOR_FALLBACKS: Dict[str, float] = {
    "Discretionary_p_optimal": 0.5,
    "Discretionary_a_th": 0.8,
    "Discretionary_b_safe": 4.5,
    "Mandatory_b_safe": 7.5,
}


def resolve_mobil_csv_column(columns: Iterable[str], canonical: str) -> Optional[str]:
    """Return the actual column name present in ``columns`` for a canonical MOBIL field."""
    for name in MOBIL_CSV_COLUMN_ALIASES.get(canonical, (canonical,)):
        if name in columns:
            return name
    return None


def mobil_params_from_csv_row(row) -> np.ndarray:
    """Build the 4-element MOBIL parameter vector from one calibration CSV row."""
    cols = row.index if hasattr(row, "index") else list(row.keys())

    def _val(canonical: str) -> float:
        col = resolve_mobil_csv_column(cols, canonical)
        if col is None:
            return float(MOBIL_VECTOR_FALLBACKS[canonical])
        return float(row[col])

    return np.array(
        [_val(c) for c in MOBIL_VECTOR_COLUMNS],
        dtype=np.float64,
    )


def load_mobil_results_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load MOBIL calibration table; empty frame if missing."""
    p = path or MOBIL_RESULTS_PATH
    try:
        df = pd.read_csv(str(p))
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def mobil_default_stats_from_csv(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Mean/Std per GUI MOBIL key from MOBIL_results.csv (same pattern as IDM/PT defaults).
    """
    stats: Dict[str, Dict[str, float]] = {}
    for gui_key, canonical in MOBIL_GUI_TO_CSV.items():
        stats[gui_key] = {
            "Mean": float(MOBIL_VECTOR_FALLBACKS[canonical]),
            "Std": 0.0,
        }

    df = load_mobil_results_csv(path)
    if df.empty:
        return stats

    cols = list(df.columns)
    for gui_key, canonical in MOBIL_GUI_TO_CSV.items():
        col = resolve_mobil_csv_column(cols, canonical)
        if col is None:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        stats[gui_key] = {"Mean": float(s.mean()), "Std": std}

    return stats
