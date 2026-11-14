"""
edge_benchmark.py — Etapa 3 (bloque 7 del diagrama metodologico).

Mide el costo operacional de cada modelo del catalogo (outputs/models/index.json)
y marca cuales cumplen las restricciones de borde. Esta pensado para
EJECUTARSE EN LA RASPBERRY PI 5 (copiar la carpeta outputs/models y data/splits).
Los modelos se entrenaron en el equipo de desarrollo; aqui solo hay INFERENCIA.
El Autoencoder se ejecuta en NumPy puro: no hace falta TensorFlow en la Pi.

Cada modelo se mide en un PROCESO NUEVO con un solo hilo numerico, de modo que:
  * la RAM refleja el modelo y no restos de modelos anteriores;
  * el CPU no depende de cuantos nucleos tenga la maquina.

Se registra (lote de 1 muestra, como llegaria un flujo real):
  latency_median_ms / latency_p95_ms / latency_mean_ms
  cpu_ms_per_inference   tiempo de CPU consumido por inferencia
  cpu_percent            (heredado) % del total de nucleos en bucle continuo
  cpu_util_at_rate       % de la CPU total al ritmo edge.flow_rate_per_s
  ram_peak_mb            pico de memoria residente del proceso
  ram_delta_mb           aumento de RAM al cargar el modelo
  model_size_kb          tamano del artefacto en disco

Cumplimiento: latencia (median o p95 segun edge.latency_stat) < max_latency_ms,
cpu_util_at_rate < max_cpu_percent, ram_peak_mb < max_ram_mb.

USO (en la Raspberry)
    python src/edge_benchmark.py --format csv
"""

from __future__ import annotations

import os

# Un solo hilo numerico: DEBE fijarse antes de importar numpy/scikit-learn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse             # noqa: E402
import json                 # noqa: E402
import pickle               # noqa: E402
import subprocess           # noqa: E402
import sys                  # noqa: E402
import time                 # noqa: E402
from pathlib import Path    # noqa: E402

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import psutil               # noqa: E402

from common import load_config, load_split, outputs_dir, save_json   # noqa: E402


def device_info() -> dict:
    model = ""
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except Exception:
        pass
    return {"device_model": model or "desconocido",
            "is_raspberry_pi": "raspberry" in model.lower(),
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2)}


def worker(model_path: str, cfg_path: str | None, fmt: str, n_runs: int, n_warmup: int) -> dict:
    """Se ejecuta en un proceso aparte. Imprime un JSON con las mediciones."""
    import resource
    cfg = load_config(cfg_path)
    X, _ = load_split(cfg, "val", fmt)              # val: sin etiquetas
    X = X[: max(2000, n_runs)]
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    with open(model_path, "rb") as fh:
        det = pickle.load(fh)
    rss_loaded = proc.memory_info().rss
    rng = np.random.default_rng(42)
    for _ in range(n_warmup):
        i = int(rng.integers(0, len(X)))
        det.scores(X[i:i + 1])
    lat = np.empty(n_runs)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for k in range(n_runs):
        i = int(rng.integers(0, len(X)))
        t0 = time.perf_counter_ns()
        det.scores(X[i:i + 1])
        lat[k] = (time.perf_counter_ns() - t0) / 1e6
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # KB -> MB (Linux)
    ncpu = psutil.cpu_count() or 1
    cpu_ms = cpu / n_runs * 1000.0
    rate = float(cfg["edge"]["flow_rate_per_s"])
    return {
        "latency_median_ms": round(float(np.median(lat)), 4),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 4),
        "latency_mean_ms": round(float(lat.mean()), 4),
        "cpu_ms_per_inference": round(cpu_ms, 5),
        "cpu_percent": round(100.0 * cpu / max(wall, 1e-9) / ncpu, 2),
        "cpu_util_at_rate": round(cpu_ms * rate / 1000.0 / ncpu * 100.0, 3),
        "ram_peak_mb": round(peak, 2),
        "ram_delta_mb": round((rss_loaded - rss0) / 1024 ** 2, 2),
        "model_size_kb": round(os.path.getsize(model_path) / 1024.0, 2),
        "n_runs": n_runs,
    }


def meets(cfg, r: dict) -> dict:
    e = cfg["edge"]
    lat = r["latency_p95_ms"] if e["latency_stat"] == "p95" else r["latency_median_ms"]
    return {"cumple_latencia": bool(lat < e["max_latency_ms"]),
            "cumple_cpu": bool(r["cpu_util_at_rate"] < e["max_cpu_percent"]),
            "cumple_ram": bool(r["ram_peak_mb"] < e["max_ram_mb"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    edge = cfg["edge"]
    n_runs = args.runs or edge["n_inference_runs"]
    n_warm = args.warmup if args.warmup is not None else edge["n_warmup_runs"]

    if args.worker:
        print("RESULT " + json.dumps(worker(args.worker, args.config, args.format, n_runs, n_warm)))
        return

    info = device_info()
    print(f"Plataforma: {info['device_model']} | {info['cpu_count']} nucleos | "
          f"{info['total_ram_gb']} GB RAM")
    if not info["is_raspberry_pi"]:
        print("[!] AVISO: este equipo NO es una Raspberry Pi. Los resultados sirven para "
              "probar el pipeline, no para reportar costo en el borde.")

    index_path = outputs_dir(cfg) / "models" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"

    rows = []
    for k, e in enumerate(index):
        tag = f"{e['detector']}/{e['metric']}/seed{e['seed']}/rank{e['rank']}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               str(index_path.parent / e["file"]), "--format", args.format,
               "--runs", str(n_runs), "--warmup", str(n_warm)]
        if args.config:
            cmd += ["--config", args.config]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if p.returncode != 0 or line is None:
            print(f"[{k + 1}/{len(index)}] {tag}  [!] {p.stderr.strip()[-200:]}")
            continue
        res = json.loads(line[7:])
        row = {k2: v for k2, v in e.items() if k2 not in ("params", "file")} | res | meets(cfg, res)
        row["platform"] = info["device_model"]
        rows.append(row)
        print(f"[{k + 1}/{len(index)}] {tag}: mediana {res['latency_median_ms']:.3f} ms | "
              f"p95 {res['latency_p95_ms']:.3f} | CPU@{edge['flow_rate_per_s']:.0f}/s "
              f"{res['cpu_util_at_rate']:.2f}% | RAM {res['ram_peak_mb']:.0f} MB")

    if not rows:
        raise RuntimeError("No se pudo medir ningun modelo.")
    out = outputs_dir(cfg) / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "edge_benchmark.csv", index=False)
    save_json({**info, "max_latency_ms": edge["max_latency_ms"],
               "max_cpu_percent": edge["max_cpu_percent"], "max_ram_mb": edge["max_ram_mb"],
               "flow_rate_per_s": edge["flow_rate_per_s"], "latency_stat": edge["latency_stat"]},
              out / "edge_platform.json")
    print(f"\nResultados en {out / 'edge_benchmark.csv'}")


if __name__ == "__main__":
    main()
