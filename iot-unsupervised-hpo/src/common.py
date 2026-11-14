"""
common.py — utilidades compartidas por todas las etapas del pipeline.

Centraliza: configuracion (con valores por defecto para las claves nuevas, de
modo que un experiment.yaml antiguo siga funcionando), lectura/escritura de
tablas, semillas, particion calibracion/objetivo del bloque de validacion,
y metricas de evaluacion (supervisadas, solo para evaluacion final y oraculo).
"""

from __future__ import annotations

import copy
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
# Valores por defecto de las claves NUEVAS (el yaml del usuario siempre gana)
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Any] = {
    "detectors": {
        "iforest": ["silhouette", "davies_bouldin"],
        "lof": ["silhouette", "davies_bouldin"],
        "autoencoder": ["reconstruction", "stability"],
    },
    # Regla de decision: umbral = percentil q de los puntajes de trafico normal
    # de CALIBRACION. q NO es un hiperparametro: se fija a priori.
    "threshold": {
        "q": 0.95,                       # tasa objetivo de falsas alarmas = 1 - q
        "sensitivity_q": [0.90, 0.95, 0.99],   # solo para reportar, nunca para elegir
        "calib_frac": 0.5,               # fraccion de `val` usada para calibrar
        "oracle_q_range": [0.80, 0.999], # rango de q en el oraculo "oracle_q"
    },
    "split": {
        "seed": 42,
        "train_frac": 0.70,
        "val_frac": 0.15,
        "attack_frac_in_test": 1.0,
        "oracle_frac_of_test": 0.5,      # parte del bloque de prueba para el oraculo
        "refit_scaler": True,            # reescala con media/desv. del train normal
        "report_prevalences": [0.05, 0.10],  # MCC ajustado a prevalencia realista
    },
    "optimization": {
        "n_trials": 100,
        "seeds": [42, 43, 44, 45, 46],
        "top_k": 5,                      # configuraciones que pasan a la Raspberry
        "oracle": True,
        "max_train_rows": None,
        "max_val_rows": None,
        "max_train_rows_lof": 20000,     # LOF guarda todo el train: se acota por borde
        "silhouette_sample": 5000,
        "consistency_boot": 3,
        "consistency_frac": 0.7,
        "latency_samples": 50,
    },
    "objectives": {"stability_replicas": 10, "stability_noise_sigma": 0.01},
    "search_space": {"iforest_max_samples": "absolute"},   # absolute | fraction
    "autoencoder": {"backend": "auto"},                    # auto | keras | sklearn
    "edge": {
        "max_latency_ms": 100.0,
        "max_cpu_percent": 50.0,
        "max_ram_mb": 512.0,
        "max_model_kb": 51200.0,
        "n_inference_runs": 1000,
        "n_warmup_runs": 20,
        "flow_rate_per_s": 100.0,        # flujos/s con que se evalua la carga de CPU
        "latency_stat": "p95",           # median | p95
        "dev_slowdown_factor": 5.0,      # Pi 5 ~5x mas lenta que el equipo de desarrollo
    },
}


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULTS, user)
    if "preprocessor_root" in cfg.get("paths", {}):
        cfg["paths"]["preprocessor_root"] = str(
            Path(os.path.expanduser(cfg["paths"]["preprocessor_root"])))
    cfg["_config_path"] = str(cfg_path)
    return cfg


def selector_paths(cfg, ext: str = "parquet") -> tuple[Path, Path]:
    base = (Path(cfg["paths"]["preprocessor_root"]) / "Datasets" / "Parquet"
            / cfg["paths"]["dataset_name"] / cfg["paths"]["selector"])
    return (base / "train" / f"best_features.{ext}",
            base / "test" / f"best_features.{ext}")


def splits_dir(cfg) -> Path:
    return REPO_ROOT / cfg["paths"]["splits_dir"]


def outputs_dir(cfg) -> Path:
    return REPO_ROOT / cfg["paths"]["outputs_dir"]


# --------------------------------------------------------------------------- #
# Entrada / salida
# --------------------------------------------------------------------------- #
def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".gz"):
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path.suffix}")


def write_table(df: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def sha256_of(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj: Any, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def limit_threads(n: int = 1) -> None:
    """Fuerza un hilo en las librerias numericas (debe llamarse antes de importar numpy)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


# --------------------------------------------------------------------------- #
# Particiones
# --------------------------------------------------------------------------- #
def load_split(cfg, name: str, fmt: str = "parquet"):
    """
    Devuelve (X, y). y es None en train/val (sin etiquetas por protocolo).
    Bloques con etiquetas: test_oracle (solo oraculo) y test_final (solo evaluate.py).
    """
    df = read_table(splits_dir(cfg) / f"{name}.{fmt}")
    if "y_true" in df.columns:
        y = df["y_true"].to_numpy()
        X = df.drop(columns=["y_true"]).to_numpy(dtype=np.float32)
        return X, y
    return df.to_numpy(dtype=np.float32), None


def split_val(X_val: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide el bloque de validacion (solo normal, sin etiquetas) en dos mitades
    disjuntas y deterministas:
        calibracion -> fija el umbral (percentil q)
        objetivo    -> se calculan las funciones objetivo
    Separarlas evita que el umbral y el objetivo se calculen sobre las mismas
    filas (circularidad) y evita el sesgo de puntuar con LOF los mismos datos
    con que se ajusto.
    """
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    idx = rng.permutation(len(X_val))
    n_cal = max(int(round(cfg["threshold"]["calib_frac"] * len(X_val))), 1)
    return X_val[idx[:n_cal]], X_val[idx[n_cal:]]


def subsample(X: np.ndarray, max_rows, seed: int = 42) -> np.ndarray:
    if max_rows is None or len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), size=int(max_rows), replace=False)]


def train_for(cfg, detector: str, X_train: np.ndarray) -> np.ndarray:
    """Train efectivo de cada detector (LOF se acota: guarda todo el train en RAM)."""
    opt = cfg["optimization"]
    X = subsample(X_train, opt.get("max_train_rows"))
    if detector == "lof":
        X = subsample(X, opt.get("max_train_rows_lof"))
    return X


# --------------------------------------------------------------------------- #
# Metricas (solo para oraculo y evaluacion final; las usa evaluate.py)
# --------------------------------------------------------------------------- #
def mcc_from_rates(tpr: float, fpr: float, prevalence: float) -> float:
    """MCC esperado con una prevalencia de ataque dada, a partir de TPR y FPR.
    TPR y FPR no dependen de la prevalencia, asi que permite reportar el MCC
    que se obtendria con trafico de ataque minoritario sin descartar datos."""
    p, n = prevalence, 1.0 - prevalence
    tp, fn, fp, tn = p * tpr, p * (1 - tpr), n * fpr, n * (1 - fpr)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def supervised_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       prevalences=()) -> Dict[str, float]:
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    out = {
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "BACC": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "FPR": float(fpr),
        "FNR": float(1 - tpr) if (fn + tp) else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    for pi in prevalences:
        out[f"MCC_pi{int(round(pi * 100)):02d}"] = mcc_from_rates(tpr, fpr, pi)
    return out


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Metricas que no dependen del umbral."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y_true)) < 2:
        return {"AUROC": float("nan"), "AUPRC": float("nan")}
    return {"AUROC": float(roc_auc_score(y_true, scores)),
            "AUPRC": float(average_precision_score(y_true, scores))}
