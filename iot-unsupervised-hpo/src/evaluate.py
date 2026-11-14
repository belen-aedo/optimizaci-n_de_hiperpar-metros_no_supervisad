"""
evaluate.py — Etapa 4 (bloques 8 y 9 del diagrama metodologico).

UNICO script autorizado a abrir test_final (y sus etiquetas). Se ejecuta una
vez, con las configuraciones ya fijadas.

Seleccion de la configuracion (SIN etiquetas)
---------------------------------------------
Para cada (detector, metrica, semilla), optimize.py deja las k mejores
configuraciones ordenadas por SU PROPIA funcion objetivo. Se elige la primera
que cumple las restricciones de borde medidas en la Raspberry (latencia, CPU,
RAM; ver edge_benchmark.csv). El MCC NO interviene en la seleccion: solo se
reporta. (Elegir por MCC de test seria usar etiquetas de prueba.)
Si aun no hay edge_benchmark.csv se usa la de rango 0 y se avisa.

Escenarios que se comparan sobre test_final (mismos datos para todos):
    linea_base  hiperparametros por defecto + regla de umbral percentil q
    optimizada  criterio no supervisado (silhouette, davies_bouldin, ...)
    oraculo     MCC maximizado con las etiquetas de test_oracle
    (linea_base_nativa: decision propia de la biblioteca, solo como referencia)

Metricas: MCC en el umbral q (principal); MCC en otros q y MCC ajustado a
prevalencia (solo descriptivo); AUROC y AUPRC (no dependen del umbral).

Salidas en outputs/results/: final_evaluation.csv, selected_models.csv,
balance.csv, balance_por_metrica.csv, hypothesis_check.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from common import (load_config, load_split, outputs_dir, ranking_metrics, save_json,
                    sha256_of, splits_dir, supervised_metrics)


def verify_integrity(cfg, fmt: str) -> None:
    mp = splits_dir(cfg) / "manifest.json"
    if not mp.exists():
        print("AVISO: no hay manifiesto; se omite la verificacion de integridad.")
        return
    man = json.loads(mp.read_text(encoding="utf-8"))
    for name in ("test_oracle", "test_final"):
        exp = man.get("sha256", {}).get(name)
        if exp and exp != sha256_of(splits_dir(cfg) / f"{name}.{fmt}"):
            raise RuntimeError(f"El hash de {name} NO coincide con el manifiesto: "
                               "el bloque fue modificado; resultados no validos.")
    print("Integridad de los bloques de prueba verificada (SHA-256).")


def select_entries(index: list[dict], edge: pd.DataFrame | None) -> list[dict]:
    """Un modelo por (detector, metrica, semilla), elegido SIN etiquetas."""
    groups = defaultdict(list)
    for e in index:
        groups[(e["detector"], e["metric"], e["seed"])].append(e)
    chosen = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda e: e["rank"])
        pick, feasible = items[0], None
        if edge is not None and key[1] != "por_defecto":
            feasible = False
            for e in items:
                r = edge[(edge.detector == e["detector"]) & (edge.metric == e["metric"]) &
                         (edge.seed == e["seed"]) & (edge["rank"] == e["rank"])]
                if len(r) and bool(r.iloc[0][["cumple_latencia", "cumple_cpu", "cumple_ram"]].all()):
                    pick, feasible = e, True
                    break
        chosen.append({**pick, "selected_feasible": feasible})
    return chosen


def evaluate_entry(entry, models_dir, X, y, cfg) -> dict:
    with open(models_dir / entry["file"], "rb") as fh:
        det = pickle.load(fh)
    s = det.scores(X)
    pi = cfg["split"]["report_prevalences"]
    row = supervised_metrics(y, (s > det.threshold_).astype(int), pi)
    row.update(ranking_metrics(y, s))
    for q in cfg["threshold"]["sensitivity_q"]:
        m = supervised_metrics(y, (s > det.threshold_for(q)).astype(int))
        tag = f"q{q * 100:g}"
        row[f"MCC_{tag}"], row[f"FPR_{tag}"], row[f"Recall_{tag}"] = m["MCC"], m["FPR"], m["Recall"]
    return row, det


def summarize(df: pd.DataFrame, cfg) -> None:
    out = outputs_dir(cfg) / "results"
    base = df[df.config == "linea_base"].set_index(["detector", "seed"])
    rows, detail = [], []
    for det_name in df.detector.unique():
        for metric, g in df[(df.detector == det_name) & (df.config.isin(["optimizada", "oraculo"]))
                            ].groupby("metric"):
            g = g.set_index("seed")
            common_seeds = [s for s in g.index if (det_name, s) in base.index]
            if not common_seeds:
                continue
            d = np.array([g.loc[s, "MCC"] - base.loc[(det_name, s), "MCC"] for s in common_seeds])
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            detail.append({"detector": det_name, "metric": metric, "config": g.config.iloc[0],
                           "n_semillas": n, "mcc_medio": g.loc[common_seeds, "MCC"].mean(),
                           "mcc_std": g.loc[common_seeds, "MCC"].std(ddof=1) if n > 1 else 0.0,
                           "mcc_base_medio": np.mean([base.loc[(det_name, s), "MCC"] for s in common_seeds]),
                           "delta_mcc": d.mean(), "delta_se": se,
                           "auroc_medio": g.loc[common_seeds, "AUROC"].mean(),
                           "fpr_medio": g.loc[common_seeds, "FPR"].mean(),
                           "recall_medio": g.loc[common_seeds, "Recall"].mean()})
    det_df = pd.DataFrame(detail)
    det_df.to_csv(out / "balance_por_metrica.csv", index=False)

    balance, hyp = [], {}
    for det_name, g in det_df.groupby("detector"):
        un = g[g.config == "optimizada"]
        if un.empty:
            continue
        best = un.sort_values("mcc_medio", ascending=False).iloc[0]
        orc = g[g.metric == "oracle"]
        orc_q = g[g.metric == "oracle_q"]
        se2 = 2 * best.delta_se if np.isfinite(best.delta_se) else 0.0
        estado = ("mejora" if best.delta_mcc - se2 > 0 else
                  "empeora" if best.delta_mcc + se2 < 0 else "mantiene")
        runner = un.sort_values("mcc_medio", ascending=False).iloc[1] if len(un) > 1 else None
        margin = float(best.mcc_medio - runner.mcc_medio) if runner is not None else float("nan")
        pooled = float(np.nanmean([best.mcc_std, runner.mcc_std])) if runner is not None else float("nan")
        balance.append({
            "detector": det_name, "mcc_linea_base": round(best.mcc_base_medio, 4),
            "mejor_metrica": best.metric, "mcc_medio": round(best.mcc_medio, 4),
            "mcc_std": round(best.mcc_std, 4), "n_semillas": int(best.n_semillas),
            "delta_mcc": round(best.delta_mcc, 4),
            "mcc_oraculo": round(float(orc.mcc_medio.iloc[0]), 4) if len(orc) else None,
            "mcc_oraculo_q": round(float(orc_q.mcc_medio.iloc[0]), 4) if len(orc_q) else None,
            "estado_vs_base": estado})
        hyp[det_name] = {"mejor_metrica": best.metric, "estado_vs_base": estado,
                         "margen_sobre_segunda": margin, "dispersion_semillas": pooled,
                         "separable": bool(np.isfinite(margin) and margin > pooled)}
    if balance:
        pd.DataFrame(balance).to_csv(out / "balance.csv", index=False)
        winners = {d: v["mejor_metrica"] for d, v in hyp.items()}
        soportada = (all(v["estado_vs_base"] != "empeora" for v in hyp.values())
                     and len(set(winners.values())) > 1
                     and all(v["separable"] for v in hyp.values()))
        save_json({"hipotesis": "la mejor metrica difiere entre detectores",
                   "criterio": ("soportada solo si (1) ninguna mejor metrica empeora la linea "
                                "base, (2) las ganadoras difieren y (3) el margen sobre la "
                                "segunda supera la dispersion entre semillas"),
                   "por_detector": hyp, "metricas_ganadoras": winners,
                   "difieren": len(set(winners.values())) > 1, "hipotesis_soportada": soportada},
                  out / "hypothesis_check.json")
        print("\nMejor metrica no supervisada por detector (MCC en test_final):")
        for b in balance:
            orc = f" | oraculo {b['mcc_oraculo']:.4f}" if b["mcc_oraculo"] is not None else ""
            print(f"  {b['detector']:12s} -> {b['mejor_metrica']:15s} MCC {b['mcc_medio']:.4f} "
                  f"(base {b['mcc_linea_base']:.4f}, delta {b['delta_mcc']:+.4f}, "
                  f"{b['estado_vs_base']}){orc}")
        print(f"\nHipotesis soportada: {soportada}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    verify_integrity(cfg, args.format)
    X_te, y_te = load_split(cfg, "test_final", args.format)
    if y_te is None:
        raise RuntimeError("test_final no contiene y_true.")
    print(f"test_final: {X_te.shape} | ataques {int(y_te.sum()):,} ({y_te.mean():.1%})")

    models_dir = outputs_dir(cfg) / "models"
    index = json.loads((models_dir / "index.json").read_text(encoding="utf-8"))
    edge_path = outputs_dir(cfg) / "results" / "edge_benchmark.csv"
    edge = pd.read_csv(edge_path) if edge_path.exists() else None
    if edge is None:
        print("AVISO: no hay edge_benchmark.csv; se usa la config de rango 0 (sin filtro de borde).")
    else:
        required = {"detector", "metric", "seed", "rank", "cumple_latencia",
                    "cumple_cpu", "cumple_ram"}
        missing = required.difference(edge.columns)
        if missing:
            print("AVISO: edge_benchmark.csv no tiene el formato esperado "
                  f"(faltan columnas: {', '.join(sorted(missing))}); "
                  "se ignora y se usa la config de rango 0 sin filtro de borde.")
            edge = None
    if edge is not None and "platform" in edge and not edge["platform"].astype(str).str.contains("Raspberry").any():
        print("AVISO: edge_benchmark.csv NO fue medido en una Raspberry Pi.")

    chosen = select_entries(index, edge)
    rows = []
    for e in chosen:
        m, det = evaluate_entry(e, models_dir, X_te, y_te, cfg)
        rows.append({"detector": e["detector"], "config": e["config"], "metric": e["metric"],
                     "seed": e["seed"], "rank": e["rank"], "objective_value": e["value"],
                     "q": e["q"], "selected_feasible": e["selected_feasible"], **m})
        print(f"  {e['detector']:12s} {e['metric']:15s} seed={e['seed']} rank={e['rank']} "
              f"MCC={m['MCC']:.4f} AUROC={m['AUROC']:.4f}")
        if e["config"] == "linea_base":                  # referencia con la decision nativa
            pred = det.predict_native(X_te)
            mn = supervised_metrics(y_te, pred, cfg["split"]["report_prevalences"])
            mn.update(ranking_metrics(y_te, det.scores(X_te)))
            rows.append({"detector": e["detector"], "config": "linea_base_nativa",
                         "metric": "por_defecto_nativo", "seed": e["seed"], "rank": 0,
                         "objective_value": None, "q": None, "selected_feasible": None, **mn})

    out = outputs_dir(cfg) / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "final_evaluation.csv", index=False)
    pd.DataFrame([{k: v for k, v in e.items() if k != "params"} for e in chosen]
                 ).to_csv(out / "selected_models.csv", index=False)
    summarize(df, cfg)
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main()
