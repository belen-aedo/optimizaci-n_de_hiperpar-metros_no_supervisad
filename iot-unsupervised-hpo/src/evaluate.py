"""
evaluate.py — Etapa 4 (bloques 8 y 9 del diagrama metodologico).

Unico script del repositorio autorizado a leer las etiquetas del bloque de
prueba. Se ejecuta una sola vez por configuracion ya fijada.

Evalua:
    - la configuracion por defecto de cada detector  (LINEA BASE)
    - la mejor configuracion hallada por cada metrica no supervisada

y produce outputs/results/final_evaluation.csv con MCC, BACC, F1, Recall,
Precision, FPR y FNR, mas outputs/results/balance.csv con la variacion
respecto de la linea base.

USO
---
    python src/evaluate.py
    python src/evaluate.py --format csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    load_config, load_split, outputs_dir, save_json, set_seeds, sha256_of,
    splits_dir, supervised_metrics,
)
from objectives import DETECTORS


def verify_test_integrity(cfg, fmt: str) -> None:
    """Comprueba que el bloque de prueba no fue alterado durante el experimento."""
    manifest_path = splits_dir(cfg) / "manifest.json"
    if not manifest_path.exists():
        print("AVISO: no hay manifiesto; se omite la verificacion de integridad.")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256", {}).get("test")
    actual = sha256_of(splits_dir(cfg) / f"test.{fmt}")
    if expected and expected != actual:
        raise RuntimeError(
            "El hash del conjunto de prueba NO coincide con el manifiesto.\n"
            "El bloque aislado fue modificado; los resultados no son validos."
        )
    print("Integridad del conjunto de prueba verificada (SHA-256 coincide).")


def evaluate_config(
    detector_name: str,
    params: dict,
    X_train,
    X_test,
    y_test,
    batch_size: int = 2048,
) -> dict:
    Detector = DETECTORS[detector_name]
    params = dict(params)
    if detector_name in ("iforest", "lof"):
        params["n_jobs"] = 1

    det = Detector(**params).fit(X_train)
    predictions = [
        det.predict_outliers(X_test[start:start + batch_size])
        for start in range(0, len(X_test), batch_size)
    ]
    y_pred = np.concatenate(predictions) if predictions else np.empty(0, dtype=int)
    return supervised_metrics(y_test, y_pred)


def load_optuna_study(path: Path) -> tuple[str, dict, str | None, int, float | None]:
    with path.open("r", encoding="utf-8") as fh:
        st = json.load(fh)
    params = dict(st.get("best_params") or {})
    detector = st["detector"]
    seed = int(st.get("seed", 0))
    if detector in ("iforest", "autoencoder"):
        params.setdefault("random_state", seed)
    return detector, params, st.get("metric"), seed, st.get("best_value")


def config_cache_key(detector: str, params: dict) -> str:
    """Identifica ajustes idénticos para no volver a entrenar el mismo modelo."""
    effective_params = dict(params)
    if detector in ("iforest", "lof"):
        effective_params["n_jobs"] = 1
    return f"{detector}:{json.dumps(effective_params, sort_keys=True, separators=(',', ':'))}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--batch-size", type=int, default=512,
        help="Muestras por lote durante la inferencia (menor usa menos RAM).",
    )
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size debe ser un entero positivo.")

    cfg = load_config(args.config)
    set_seeds(args.seed)
    verify_test_integrity(cfg, args.format)

    X_train, _ = load_split(cfg, "train", args.format)
    X_test, y_test = load_split(cfg, "test", args.format)
    if y_test is None:
        raise RuntimeError("El conjunto de prueba no contiene la columna y_true.")
    print(f"train: {X_train.shape} | test: {X_test.shape} "
          f"| ataques en test: {int(y_test.sum()):,}")

    rows = []
    seen_configs = set()
    cached_metrics = {}

    # ---- 1. Linea base: hiperparametros por defecto ----------------------- #
    for detector_name in cfg["detectors"]:
        try:
            key = config_cache_key(detector_name, {})
            m = cached_metrics.get(key)
            if m is None:
                m = evaluate_config(
                    detector_name, {}, X_train, X_test, y_test, args.batch_size
                )
                cached_metrics[key] = m
        except Exception as exc:
            print(f"  [!] linea base {detector_name}: {exc}")
            continue
        row = {"detector": detector_name, "config": "linea_base",
               "metric": "por_defecto", "seed": args.seed, **m}
        rows.append(row)
        seen_configs.add((detector_name, "linea_base", "por_defecto", args.seed))
        print(f"  linea base {detector_name:12s} MCC={m['MCC']:.4f}")

    # ---- 2. Configuraciones optimizadas ----------------------------------- #
    studies_dir = outputs_dir(cfg) / "optuna"
    files = sorted(studies_dir.glob("*_seed*.json"))
    if not files:
        print("\nAVISO: no hay estudios en outputs/optuna. "
              "Ejecute primero src/optimize.py.")
    for f in files:
        try:
            detector, params, metric, seed, objective_value = load_optuna_study(f)
        except Exception as exc:
            print(f"  [!] {f.name}: no se pudo leer el estudio ({exc})")
            continue

        duplicate_key = (detector, "optimizada", metric, seed)
        if duplicate_key in seen_configs:
            continue
        seen_configs.add(duplicate_key)

        try:
            key = config_cache_key(detector, params)
            m = cached_metrics.get(key)
            if m is None:
                m = evaluate_config(
                    detector, params, X_train, X_test, y_test, args.batch_size
                )
                cached_metrics[key] = m
        except Exception as exc:
            print(f"  [!] {f.name}: {exc}")
            continue
        row = {
            "detector": detector, "config": "optimizada",
            "metric": metric, "seed": seed,
            "objective_value": objective_value, **m,
        }
        rows.append(row)
        print(f"  {detector:12s} {metric:15s} seed={seed} MCC={m['MCC']:.4f}")

    if not rows:
        raise RuntimeError("No se evaluo ninguna configuracion.")

    df = pd.DataFrame.from_records(rows)
    out = outputs_dir(cfg) / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "final_evaluation.csv", index=False)

    # ---- 3. Equilibrio deteccion/costo vs. linea base ---------------------- #
    balance = []
    by_detector = defaultdict(list)
    for row in rows:
        by_detector[row["detector"]].append(row)

    for detector_name, records in by_detector.items():
        base = [r for r in records if r["config"] == "linea_base"]
        opt = [r for r in records if r["config"] == "optimizada"]
        if not base or not opt:
            continue

        base_mcc = float(base[0]["MCC"])
        metric_scores = defaultdict(list)
        for row in opt:
            metric_scores[row["metric"]].append(float(row["MCC"]))

        if not metric_scores:
            continue

        best_metric, values = max(metric_scores.items(), key=lambda item: float(np.mean(item[1])))
        values_arr = np.asarray(values, dtype=np.float64)
        balance.append({
            "detector": detector_name,
            "mcc_linea_base": round(base_mcc, 4),
            "mejor_metrica": best_metric,
            "mcc_medio": round(float(values_arr.mean()), 4),
            "mcc_std": round(float(values_arr.std(ddof=1) if values_arr.size > 1 else 0.0), 4),
            "n_semillas": int(values_arr.size),
            "delta_mcc": round(float(values_arr.mean()) - base_mcc, 4),
        })

    if balance:
        pd.DataFrame(balance).to_csv(out / "balance.csv", index=False)
        print("\nMejor metrica por detector:")
        for b in balance:
            print(f"  {b['detector']:12s} -> {b['mejor_metrica']:15s} "
                  f"MCC {b['mcc_medio']:.4f} (base {b['mcc_linea_base']:.4f}, "
                  f"delta {b['delta_mcc']:+.4f})")
        save_json(
            {"hipotesis": "la mejor metrica difiere entre detectores",
             "metricas_ganadoras": {b["detector"]: b["mejor_metrica"] for b in balance},
             "difieren": len({b["mejor_metrica"] for b in balance}) > 1},
            out / "hypothesis_check.json",
        )

    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main()
