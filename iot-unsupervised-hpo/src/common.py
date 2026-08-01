"""
common.py — utilidades compartidas por todas las etapas del pipeline.

Centraliza carga de configuracion, lectura/escritura de tablas, fijacion de
semillas y calculo de metricas supervisadas, de modo que ningun script
duplique estas responsabilidades.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment.yaml"


# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Carga config/experiment.yaml y expande la ruta del preprocesador."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"]["preprocessor_root"] = str(
        Path(os.path.expanduser(cfg["paths"]["preprocessor_root"]))
    )
    cfg["_config_path"] = str(cfg_path)
    return cfg


def selector_paths(cfg: Dict[str, Any], ext: str = "parquet") -> tuple[Path, Path]:
    """Rutas a los best_features generados por preprocessor_v2."""
    base = (
        Path(cfg["paths"]["preprocessor_root"])
        / "Datasets"
        / "Parquet"
        / cfg["paths"]["dataset_name"]
        / cfg["paths"]["selector"]
    )
    return (base / "train" / f"best_features.{ext}",
            base / "test" / f"best_features.{ext}")


def splits_dir(cfg: Dict[str, Any]) -> Path:
    return REPO_ROOT / cfg["paths"]["splits_dir"]


def outputs_dir(cfg: Dict[str, Any]) -> Path:
    return REPO_ROOT / cfg["paths"]["outputs_dir"]


# --------------------------------------------------------------------------- #
# Entrada / salida
# --------------------------------------------------------------------------- #
def read_table(path: str | Path) -> pd.DataFrame:
    """Lee .parquet o .csv segun la extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}.\n"
            "Verifique 'preprocessor_root' en config/experiment.yaml y que el "
            "pipeline preprocessor_v2 se haya ejecutado para este dataset."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".gz"):
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path.suffix}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def sha256_of(path: str | Path, block: int = 1 << 20) -> str:
    """Hash del archivo, usado para verificar que el test no fue alterado."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # opcional: solo si TensorFlow esta instalado
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Carga de particiones
# --------------------------------------------------------------------------- #
def load_split(cfg: Dict[str, Any], name: str, fmt: str = "parquet"):
    """
    Devuelve (X, y) de una particion. y es None para train/val, porque esos
    bloques no tienen etiquetas disponibles en el protocolo experimental.
    """
    path = splits_dir(cfg) / f"{name}.{fmt}"
    df = read_table(path)
    target = "y_true"
    if target in df.columns:
        y = df[target].to_numpy()
        X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
        return X, y
    return df.to_numpy(dtype=np.float32), None


def subsample(X: np.ndarray, max_rows: int | None, seed: int = 42) -> np.ndarray:
    """Submuestreo aleatorio sin reemplazo para acelerar la busqueda."""
    if max_rows is None or len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx]


# --------------------------------------------------------------------------- #
# Metricas supervisadas (solo para la evaluacion final)
# --------------------------------------------------------------------------- #
def supervised_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula el panel de metricas del informe.
    Convencion: 1 = anomalia (clase positiva), 0 = normal.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "BACC": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "FPR": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "FNR": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
