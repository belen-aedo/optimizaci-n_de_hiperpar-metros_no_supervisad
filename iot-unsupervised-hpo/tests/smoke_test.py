"""
smoke_test.py — prueba de humo de punta a punta con datos SINTETICOS.

Comprueba que el pipeline corre (prepare_splits -> diagnose -> optimize ->
edge_benchmark -> evaluate) y que NO hay fuga de etiquetas:
  * optimize (escenarios no supervisados) corre con test_oracle y test_final
    ocultos;
  * el oraculo corre con test_final oculto.
Si Optuna no esta instalado usa busqueda aleatoria (ALLOW_RANDOM_FALLBACK=1):
eso valida la logica, NO reemplaza una corrida real con Optuna.

    python tests/smoke_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
rng = np.random.default_rng(0)

# ---- datos sinteticos con el formato de preprocessor_v2 (csv) --------------
d = 12
normal = rng.normal(0, 1, (4000, d))
atk_a = rng.normal(2.5, 1.2, (600, d))                       # ataque lejano
atk_b = np.hstack([rng.normal(0, 1, (400, d - 3)), rng.normal(4, 1, (400, 3))])  # parcial
X = np.vstack([normal, atk_a, atk_b])
y = np.array([0] * 4000 + [1] * 1000)
cols = [f"f{i}" for i in range(d)]
df = pd.DataFrame(X, columns=cols); df["label"] = y
idx = rng.permutation(len(df)); df = df.iloc[idx].reset_index(drop=True)
base = tmp / "pre" / "Datasets" / "Parquet" / "Synth" / "sel"
(base / "train").mkdir(parents=True); (base / "test").mkdir(parents=True)
df.iloc[:3500].to_csv(base / "train" / "best_features.csv", index=False)
df.iloc[3500:].to_csv(base / "test" / "best_features.csv", index=False)

cfg = {
    "paths": {"preprocessor_root": str(tmp / "pre"), "dataset_name": "Synth", "selector": "sel",
              "splits_dir": str(tmp / "splits"), "outputs_dir": str(tmp / "outputs")},
    "dataset": {"label_col": "label", "normal_label": "0", "metadata_cols": []},
    "split": {"seed": 42, "train_frac": 0.7, "val_frac": 0.15, "attack_frac_in_test": 1.0},
    "optimization": {"n_trials": 6, "seeds": [42, 43], "top_k": 3, "latency_samples": 10},
    "edge": {"n_inference_runs": 60, "n_warmup_runs": 5},
    "autoencoder": {"backend": "sklearn"},
    "detectors": {"iforest": ["silhouette", "davies_bouldin"], "lof": ["silhouette"],
                  "autoencoder": ["reconstruction", "stability", "consistency"]},
}
cfgp = tmp / "experiment.yaml"
cfgp.write_text(yaml.safe_dump(cfg))
env = dict(os.environ, ALLOW_RANDOM_FALLBACK="1")


def run(script, *args):
    cmd = [sys.executable, str(SRC / script), "--config", str(cfgp), *args]
    print("\n$", " ".join(["python", script, *args]))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=SRC)
    print(p.stdout[-1800:])
    if p.returncode != 0:
        print(p.stderr[-2500:])
        sys.exit(f"FALLO en {script}")


run("prepare_splits.py", "--input-format", "csv", "--format", "csv")
sp = tmp / "splits"
hidden = tmp / "hidden"; hidden.mkdir()


def hide(*names):
    for n in names:
        shutil.move(sp / f"{n}.csv", hidden / f"{n}.csv")


def show(*names):
    for n in names:
        shutil.move(hidden / f"{n}.csv", sp / f"{n}.csv")


# 1) no supervisado: ni test_oracle ni test_final existen -> no pueden leerse
hide("test_oracle", "test_final")
run("optimize.py", "--all", "--no-oracle", "--format", "csv", "--trials", "6", "--seeds", "42", "43")
# 2) oraculo: solo test_oracle visible
show("test_oracle")
run("optimize.py", "--detector", "iforest", "--metric", "oracle", "--format", "csv",
    "--trials", "6", "--seeds", "42", "43")
run("optimize.py", "--detector", "lof", "--metric", "oracle_q", "--format", "csv",
    "--trials", "6", "--seeds", "42", "43")
run("diagnose_objectives.py", "--n", "8", "--format", "csv")
show("test_final")
print("\n>>> LEAK CHECK OK: optimize no necesito test_final ni (salvo oraculo) test_oracle")

run("edge_benchmark.py", "--format", "csv")
run("evaluate.py", "--format", "csv")

res = tmp / "outputs" / "results"
print("\nARCHIVOS:", sorted(p.name for p in res.glob("*")))
print(pd.read_csv(res / "balance.csv").to_string())
print("\nSMOKE TEST OK ->", tmp)
