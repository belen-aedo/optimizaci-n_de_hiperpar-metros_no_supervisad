"""
edge_benchmark.py — Etapa 3 (bloque 7 del diagrama metodologico).

Mide el costo operacional real de cada configuracion sobre el dispositivo de
borde (Raspberry Pi 5, 8 GB). Este script esta pensado para EJECUTARSE EN LA
RASPBERRY, no en la maquina de desarrollo.

Se registran:
    - tiempo de inferencia por muestra: mediana y percentil 95 (ms)
    - uso medio de CPU durante la ventana de inferencia (%)
    - peak de memoria residente del proceso (MB)
    - tamano del modelo serializado (KB)

La distribucion de latencias es asimetrica, por lo que se reporta la mediana
en lugar de la media, acompanada del percentil 95.

USO (en la Raspberry Pi)
------------------------
    python src/edge_benchmark.py --format csv
    python src/edge_benchmark.py --runs 2000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from common import load_config, load_split, outputs_dir, save_json, set_seeds
from objectives import DETECTORS


def model_size_kb(detector) -> float:
    """Tamano en disco del artefacto serializado."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fh:
            pickle.dump(getattr(detector, "model", detector), fh)
            path = fh.name
        size = os.path.getsize(path) / 1024.0
        os.unlink(path)
        return round(size, 2)
    except Exception:
        return float("nan")


def benchmark(detector, X: np.ndarray, n_runs: int, n_warmup: int) -> dict:
    proc = psutil.Process(os.getpid())
    rng = np.random.default_rng(42)

    # Calentamiento: descarta el costo de la primera llamada y de la cache fria
    for _ in range(n_warmup):
        i = int(rng.integers(0, len(X)))
        detector.scores(X[i:i + 1])

    proc.cpu_percent(interval=None)          # inicializa el contador
    rss_before = proc.memory_info().rss
    peak_rss = rss_before
    latencies_ns = np.empty(n_runs, dtype=np.int64)

    for k in range(n_runs):
        i = int(rng.integers(0, len(X)))
        sample = X[i:i + 1]
        t0 = time.perf_counter_ns()
        detector.scores(sample)
        latencies_ns[k] = time.perf_counter_ns() - t0
        if k % 50 == 0:
            peak_rss = max(peak_rss, proc.memory_info().rss)

    # cpu_percent puede superar 100 si las librerias de algebra usan varios
    # hilos; se reporta el valor crudo y el normalizado por numero de nucleos.
    cpu_raw = proc.cpu_percent(interval=None)
    cpu = cpu_raw / max(psutil.cpu_count() or 1, 1)
    lat_ms = latencies_ns / 1e6
    return {
        "latency_median_ms": round(float(np.median(lat_ms)), 4),
        "latency_p95_ms": round(float(np.percentile(lat_ms, 95)), 4),
        "latency_mean_ms": round(float(np.mean(lat_ms)), 4),
        "cpu_percent": round(float(cpu), 2),          # % del total de nucleos
        "cpu_percent_raw": round(float(cpu_raw), 2),  # % de un nucleo
        "ram_peak_mb": round(peak_rss / 1024 ** 2, 2),
        "model_size_kb": model_size_kb(detector),
        "n_runs": n_runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    edge = cfg["edge"]
    n_runs = args.runs or edge["n_inference_runs"]
    n_warmup = args.warmup if args.warmup is not None else edge["n_warmup_runs"]
    set_seeds(42)

    X_train, _ = load_split(cfg, "train", args.format)
    X_val, _ = load_split(cfg, "val", args.format)
    print(f"Plataforma: {psutil.cpu_count()} nucleos | "
          f"{psutil.virtual_memory().total / 1024**3:.1f} GB RAM")

    configs = [{"detector": d, "config": "linea_base", "metric": "por_defecto",
                "seed": 42, "params": {}} for d in cfg["detectors"]]

    for f in sorted((outputs_dir(cfg) / "optuna").glob("*_seed*.json")):
        st = json.loads(f.read_text(encoding="utf-8"))
        params = dict(st["best_params"])
        if st["detector"] in ("iforest", "autoencoder"):
            params.setdefault("random_state", st["seed"])
        configs.append({"detector": st["detector"], "config": "optimizada",
                        "metric": st["metric"], "seed": st["seed"], "params": params})

    rows = []
    for c in configs:
        tag = f"{c['detector']}/{c['metric']}/seed{c['seed']}"
        print(f"\n-> {tag}")
        try:
            det = DETECTORS[c["detector"]](**c["params"]).fit(X_train)
            res = benchmark(det, X_val, n_runs, n_warmup)
        except Exception as exc:
            print(f"   [!] {exc}")
            continue
        res["cumple_latencia"] = bool(res["latency_median_ms"] < edge["max_latency_ms"])
        res["cumple_cpu"] = bool(res["cpu_percent"] < edge["max_cpu_percent"])
        rows.append({k: v for k, v in c.items() if k != "params"} | res)
        print(f"   mediana {res['latency_median_ms']:.3f} ms | "
              f"p95 {res['latency_p95_ms']:.3f} ms | "
              f"CPU {res['cpu_percent']:.1f}% | RAM {res['ram_peak_mb']:.1f} MB")

    if not rows:
        raise RuntimeError("No se pudo medir ninguna configuracion.")

    out = outputs_dir(cfg) / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "edge_benchmark.csv", index=False)
    save_json(
        {"cpu_count": psutil.cpu_count(),
         "total_ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
         "max_latency_ms": edge["max_latency_ms"],
         "max_cpu_percent": edge["max_cpu_percent"]},
        out / "edge_platform.json",
    )
    print(f"\nResultados en {out / 'edge_benchmark.csv'}")


if __name__ == "__main__":
    main()
