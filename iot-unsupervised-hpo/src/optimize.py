"""
optimize.py — Etapa 2 (bloques 5 y 6 del diagrama metodologico).

Ejecuta la busqueda de hiperparametros con Optuna usando EXCLUSIVAMENTE
funciones objetivo no supervisadas, calculadas sobre el bloque de validacion
que contiene solo trafico normal. El bloque de prueba nunca se abre aqui.

Formulacion
-----------
Cada metrica se optimiza como objetivo unico. Las restricciones de hardware se
aplican como restriccion dura: un trial cuya latencia estimada supera el
presupuesto se descarta mediante optuna.TrialPruned, de modo que la restriccion
no altera la escala del objetivo ni mezcla su efecto con el del criterio no
supervisado (ver justificacion en la seccion de decisiones metodologicas).

USO
---
    python src/optimize.py --detector iforest --metric silhouette
    python src/optimize.py --detector lof --metric davies_bouldin --trials 50
    python src/optimize.py --all                 # todas las combinaciones
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import load_config, load_split, outputs_dir, save_json, set_seeds, subsample
from objectives import DETECTORS, DIRECTION, compute_objective, penalty_for


# --------------------------------------------------------------------------- #
# Espacios de busqueda (informe, Tabla IV)
# --------------------------------------------------------------------------- #
def suggest_params(trial, detector: str, seed: int) -> dict:
    if detector == "iforest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "contamination": trial.suggest_float("contamination", 1e-3, 0.1, log=True),
            "n_jobs": -1,
            "random_state": seed,
        }
    if detector == "lof":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 100),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
            "contamination": trial.suggest_float("contamination", 1e-3, 0.1, log=True),
            "n_jobs": -1,
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


def measure_latency_ms(detector, X: np.ndarray, n: int = 200) -> float:
    """
    Latencia media de inferencia por muestra en la maquina de desarrollo.
    Es un filtro preliminar: la medicion definitiva se hace sobre la
    Raspberry Pi 5 en edge_benchmark.py.
    """
    sample = X[:n] if len(X) > n else X
    t0 = time.perf_counter_ns()
    detector.scores(sample)
    elapsed_ns = time.perf_counter_ns() - t0
    return (elapsed_ns / len(sample)) / 1e6


# --------------------------------------------------------------------------- #
# Busqueda
# --------------------------------------------------------------------------- #
def run_study(cfg, detector_name: str, metric: str, seed: int,
              n_trials: int, X_train, X_val) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    set_seeds(seed)
    direction = DIRECTION[metric]
    budget = float(cfg["optimization"]["dev_latency_budget_ms"])
    Detector = DETECTORS[detector_name]

    def objective(trial):
        params = suggest_params(trial, detector_name, seed)
        try:
            det = Detector(**params).fit(X_train)
        except Exception as exc:
            trial.set_user_attr("error", str(exc)[:200])
            raise optuna.TrialPruned()

        lat = measure_latency_ms(det, X_val)
        trial.set_user_attr("latency_ms", lat)
        if lat > budget:
            # Restriccion dura: configuracion inviable en el borde
            trial.set_user_attr("pruned_reason", "latency")
            raise optuna.TrialPruned()

        value = compute_objective(metric, det, X_val, cfg)
        if not np.isfinite(value):
            return penalty_for(metric)
        return value

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner,
        study_name=f"{detector_name}_{metric}_seed{seed}",
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    elapsed = time.perf_counter() - t0

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise RuntimeError(
            f"Ningun trial completo para {detector_name}/{metric}. "
            "Revise el presupuesto de latencia o el espacio de busqueda."
        )

    return {
        "detector": detector_name,
        "metric": metric,
        "direction": direction,
        "seed": seed,
        "n_trials": n_trials,
        "n_completed": len(completed),
        "n_pruned": len(study.trials) - len(completed),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_latency_ms": study.best_trial.user_attrs.get("latency_ms"),
        "elapsed_s": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--detector", choices=list(DETECTORS))
    ap.add_argument("--metric", choices=list(DIRECTION))
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--fast", action="store_true",
                    help="Prueba rapida: 25 trials y una semilla (42)")
    ap.add_argument("--all", action="store_true",
                    help="Ejecuta todas las combinaciones de config['detectors']")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_trials = args.trials or (25 if args.fast else cfg["optimization"]["n_trials"])
    seeds = (args.seeds if args.seeds is not None
             else ([42] if args.fast else cfg["optimization"]["seeds"]))

    X_train, _ = load_split(cfg, "train", args.format)
    X_val, _ = load_split(cfg, "val", args.format)
    X_train = subsample(X_train, cfg["optimization"].get("max_train_rows"))
    X_val = subsample(X_val, cfg["optimization"].get("max_val_rows"))
    print(f"train: {X_train.shape} | val: {X_val.shape}")

    if args.all:
        combos = [(d, m) for d, ms in cfg["detectors"].items() for m in ms]
    else:
        if not (args.detector and args.metric):
            ap.error("Indique --detector y --metric, o use --all")
        combos = [(args.detector, args.metric)]

    out_dir = outputs_dir(cfg) / "optuna"
    results = []
    for detector, metric in combos:
        for seed in seeds:
            tag = f"{detector}_{metric}_seed{seed}"
            print(f"\n=== {tag} ({n_trials} trials) ===")
            res = run_study(cfg, detector, metric, seed, n_trials, X_train, X_val)
            save_json(res, out_dir / f"{tag}.json")
            results.append(res)
            print(f"  mejor {metric}: {res['best_value']:.4f} "
                  f"| completos {res['n_completed']} | podados {res['n_pruned']} "
                  f"| {res['elapsed_s']}s")

    save_json(results, out_dir / "all_studies.json")
    print(f"\nResultados en {out_dir}")


if __name__ == "__main__":
    main()
