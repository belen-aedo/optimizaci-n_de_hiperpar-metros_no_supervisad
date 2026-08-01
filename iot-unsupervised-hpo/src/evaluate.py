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


def evaluate_config(detector_name: str, params: dict, X_train, X_test, y_test) -> dict:
    Detector = DETECTORS[detector_name]
    det = Detector(**params).fit(X_train)
    y_pred = det.predict_outliers(X_test)
    return supervised_metrics(y_test, y_pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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

    # ---- 1. Linea base: hiperparametros por defecto ----------------------- #
    for detector_name in cfg["detectors"]:
        try:
            m = evaluate_config(detector_name, {}, X_train, X_test, y_test)
        except Exception as exc:
            print(f"  [!] linea base {detector_name}: {exc}")
            continue
        rows.append({"detector": detector_name, "config": "linea_base",
                     "metric": "por_defecto", "seed": args.seed, **m})
        print(f"  linea base {detector_name:12s} MCC={m['MCC']:.4f}")

    # ---- 2. Configuraciones optimizadas ----------------------------------- #
    studies_dir = outputs_dir(cfg) / "optuna"
    files = sorted(studies_dir.glob("*_seed*.json"))
    if not files:
        print("\nAVISO: no hay estudios en outputs/optuna. "
              "Ejecute primero src/optimize.py.")
    for f in files:
        st = json.loads(f.read_text(encoding="utf-8"))
        params = dict(st["best_params"])
        if st["detector"] in ("iforest", "autoencoder"):
            params.setdefault("random_state", st["seed"])
        try:
            m = evaluate_config(st["detector"], params, X_train, X_test, y_test)
        except Exception as exc:
            print(f"  [!] {f.name}: {exc}")
            continue
        rows.append({
            "detector": st["detector"], "config": "optimizada",
            "metric": st["metric"], "seed": st["seed"],
            "objective_value": st["best_value"], **m,
        })
        print(f"  {st['detector']:12s} {st['metric']:15s} seed={st['seed']} "
              f"MCC={m['MCC']:.4f}")

    if not rows:
        raise RuntimeError("No se evaluo ninguna configuracion.")

    df = pd.DataFrame(rows)
    out = outputs_dir(cfg) / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "final_evaluation.csv", index=False)

    # ---- 3. Equilibrio deteccion/costo vs. linea base ---------------------- #
    balance = []
    for detector_name, grp in df.groupby("detector"):
        base = grp[grp["config"] == "linea_base"]
        opt = grp[grp["config"] == "optimizada"]
        if base.empty or opt.empty:
            continue
        base_mcc = float(base["MCC"].iloc[0])
        agg = opt.groupby("metric")["MCC"].agg(["mean", "std", "count"])
        best_metric = agg["mean"].idxmax()
        balance.append({
            "detector": detector_name,
            "mcc_linea_base": round(base_mcc, 4),
            "mejor_metrica": best_metric,
            "mcc_medio": round(float(agg.loc[best_metric, "mean"]), 4),
            "mcc_std": round(float(agg.loc[best_metric, "std"] or 0.0), 4),
            "n_semillas": int(agg.loc[best_metric, "count"]),
            "delta_mcc": round(float(agg.loc[best_metric, "mean"]) - base_mcc, 4),
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
