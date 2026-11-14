"""
diagnose_objectives.py — ¿sirve cada metrica no supervisada como guia?

ANTES de gastar horas en Optuna: se toman N configuraciones aleatorias por
detector, y para cada una se mide (a) cada funcion objetivo no supervisada
sobre el bloque `objetivo` de val, y (b) la calidad REAL de deteccion (MCC y
AUROC) sobre test_oracle. Se reporta la correlacion de Spearman entre ambas.

  rho > 0 (con signo ajustado)  -> la metrica ordena bien las configuraciones
  rho ~ 0                       -> no informa: optimizarla es optimizar ruido
  rho < 0                       -> las empeora

Usa etiquetas SOLO de test_oracle (nunca de test_final) y solo como
diagnostico: no interviene en ninguna seleccion. Tambien sirve como analisis
para el capitulo de resultados (relacion criterio -> desempeno).

USO
    python src/diagnose_objectives.py --n 40 --format csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (load_config, load_split, outputs_dir, ranking_metrics, save_json,
                    split_val, supervised_metrics, train_for)
from objectives import DETECTORS, DIRECTION, UNSUPERVISED, compute_objective
from search_space import RandomTrial, suggest_params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--n", type=int, default=40, help="configuraciones por detector")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    X_train, _ = load_split(cfg, "train", args.format)
    X_val, _ = load_split(cfg, "val", args.format)
    X_orc, y_orc = load_split(cfg, "test_oracle", args.format)
    X_cal, X_obj = split_val(X_val, cfg)
    metrics = args.metrics or list(UNSUPERVISED)

    recs = []
    for dname, Detector in DETECTORS.items():
        X_tr = train_for(cfg, dname, X_train)
        print(f"\n{dname}: {args.n} configuraciones aleatorias")
        for k in range(args.n):
            trial = RandomTrial(args.seed * 100000 + k)
            params = suggest_params(trial, dname, args.seed + k, cfg)
            try:
                det = Detector(**params).fit(X_tr).set_threshold(X_cal, cfg["threshold"]["q"])
            except Exception as exc:
                print("  [!]", exc)
                continue
            s = det.scores(X_orc)
            r = {"detector": dname, "trial": k, **trial.params,
                 "MCC": supervised_metrics(y_orc, (s > det.threshold_).astype(int))["MCC"],
                 **ranking_metrics(y_orc, s)}
            for m in metrics:
                try:
                    r[m] = compute_objective(m, det, X_obj, cfg, ctx={
                        "Detector": Detector, "params": params, "X_train": X_tr})
                except Exception:
                    r[m] = np.nan
            recs.append(r)

    df = pd.DataFrame(recs)
    out = outputs_dir(cfg) / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "diagnose_objectives.csv", index=False)

    rows = []
    print("\nCorrelacion de Spearman (signo ajustado: + = la metrica ordena bien)")
    print(f"{'detector':12s} {'metrica':15s} {'rho vs MCC':>11s} {'rho vs AUROC':>13s} {'n':>4s}")
    for dname, g in df.groupby("detector"):
        for m in metrics:
            if m not in g or g[m].nunique() < 3:
                continue
            sign = 1.0 if DIRECTION[m] == "maximize" else -1.0
            gg = g.dropna(subset=[m, "MCC", "AUROC"])
            r_mcc = sign * spearmanr(gg[m], gg["MCC"]).statistic
            r_auc = sign * spearmanr(gg[m], gg["AUROC"]).statistic
            rows.append({"detector": dname, "metric": m, "rho_mcc": r_mcc,
                         "rho_auroc": r_auc, "n": len(gg)})
            print(f"{dname:12s} {m:15s} {r_mcc:+11.3f} {r_auc:+13.3f} {len(gg):4d}")
    save_json(rows, out / "diagnose_objectives_summary.json")


if __name__ == "__main__":
    main()
