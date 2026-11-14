"""
search_space.py — espacios de busqueda (Tabla IV del informe) y un `trial`
aleatorio con la misma interfaz de Optuna, usado en diagnose_objectives.py y
en las pruebas.

CAMBIOS respecto del informe (declararlos en el capitulo 3 como enmienda):
  * `contamination` ya NO se busca. El umbral es un percentil q fijo (regla de
    decision independiente del optimizador). Era lo que producia el optimo
    degenerado.
  * iForest `max_samples`: por defecto numero ABSOLUTO de filas por arbol, en
    escala log [64, 4096] (Liu et al. recomiendan 256). Como fraccion de un
    train grande, cada arbol podia usar 100 000+ filas: modelo pesado y lento en
    el borde. Con search_space.iforest_max_samples: fraction se vuelve al rango
    original [0.1, 1.0].
"""

from __future__ import annotations

import numpy as np


class RandomTrial:
    """Imita la parte de optuna.Trial que usa suggest_params."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.params: dict = {}
        self.user_attrs: dict = {}

    def suggest_int(self, name, low, high, step=1, log=False):
        if log:
            v = int(round(np.exp(self.rng.uniform(np.log(low), np.log(high)))))
        else:
            v = low + step * int(self.rng.integers(0, (high - low) // step + 1))
        self.params[name] = int(min(max(v, low), high))
        return self.params[name]

    def suggest_float(self, name, low, high, log=False):
        v = (np.exp(self.rng.uniform(np.log(low), np.log(high))) if log
             else self.rng.uniform(low, high))
        self.params[name] = float(v)
        return self.params[name]

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[int(self.rng.integers(0, len(choices)))]
        return self.params[name]

    def set_user_attr(self, k, v):
        self.user_attrs[k] = v


def suggest_params(trial, detector: str, seed: int, cfg=None) -> dict:
    mode = ((cfg or {}).get("search_space", {}) or {}).get("iforest_max_samples", "absolute")
    if detector == "iforest":
        ms = (trial.suggest_int("max_samples", 64, 4096, log=True) if mode == "absolute"
              else trial.suggest_float("max_samples", 0.1, 1.0))
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_samples": ms,
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "random_state": seed,
        }
    if detector == "lof":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 100),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
        }
    if detector == "autoencoder":
        return {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "units": trial.suggest_int("units", 8, 128, log=True),
            "latent_dim": trial.suggest_int("latent_dim", 2, 16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "epochs": trial.suggest_int("epochs", 20, 100, step=10),
            "random_state": seed,
        }
    raise ValueError(f"Detector desconocido: {detector}")
