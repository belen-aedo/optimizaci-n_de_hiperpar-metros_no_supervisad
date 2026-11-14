"""
prepare_splits.py — Etapa 0 del pipeline (bloque 2 del diagrama metodologico).

POR QUE EXISTE ESTE SCRIPT
--------------------------
El pipeline preprocessor_v2 divide los datos en train/test 80/20 de forma
aleatoria, quedando ambos bloques con trafico mixto (normal + ataque). Ese
esquema es correcto para un benchmark supervisado, pero NO corresponde al
protocolo de esta investigacion, que exige:

    - Entrenamiento : SOLO trafico normal, sin etiquetas
    - Validacion    : SOLO trafico normal, sin etiquetas
    - Prueba        : trafico normal + ataques, etiquetado, aislado

Por lo tanto este script reune ambos bloques del preprocesador y reconstruye
la particion segun el protocolo antifuga descrito en la metodologia.

QUE PRODUCE
-----------
    data/splits/train.parquet   (features, sin etiqueta)
    data/splits/val.parquet     (features, sin etiqueta)
    data/splits/test.parquet    (features + columna y_true)
    data/splits/manifest.json   (hashes, tamanos, distribucion de clases)

USO
---
    python src/prepare_splits.py
    python src/prepare_splits.py --format csv --config config/experiment.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    load_config, outputs_dir, read_table, save_json, selector_paths,
    set_seeds, sha256_of, splits_dir, write_table,
)


def build_target(df: pd.DataFrame, cfg) -> pd.Series:
    """Devuelve una serie binaria: 1 = anomalia, 0 = normal."""
    label_col = cfg["dataset"]["label_col"]
    normal = str(cfg["dataset"]["normal_label"]).lower()

    if label_col not in df.columns:
        raise KeyError(
            f"La columna de etiqueta '{label_col}' no esta en los datos.\n"
            f"Columnas disponibles: {list(df.columns)[:20]} ...\n"
            "Ajuste dataset.label_col en config/experiment.yaml."
        )
    raw = df[label_col].astype(str).str.strip().str.lower()
    return (raw != normal).astype(int)


def feature_matrix(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Descarta etiqueta y metadatos, y conserva solo columnas numericas."""
    drop = [cfg["dataset"]["label_col"], *cfg["dataset"]["metadata_cols"]]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"],
                    help="Formato de salida de las particiones")
    ap.add_argument("--input-format", default="parquet", choices=["parquet", "csv"],
                    help="Formato de los best_features de preprocessor_v2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp = cfg["split"]
    set_seeds(sp["seed"])

    # ---- 1. Cargar las salidas del preprocesador y reunirlas -------------- #
    train_path, test_path = selector_paths(cfg, args.input_format)
    print(f"Leyendo espacio de caracteristicas '{cfg['paths']['selector']}':")
    print(f"  {train_path}")
    print(f"  {test_path}")
    df = pd.concat([read_table(train_path), read_table(test_path)], ignore_index=True)
    print(f"-> {len(df):,} filas reunidas, {df.shape[1]} columnas")

    y = build_target(df, cfg)
    X = feature_matrix(df, cfg)
    print(f"-> {X.shape[1]} caracteristicas numericas conservadas")
    print(f"-> normal: {(y == 0).sum():,} | ataque: {(y == 1).sum():,}")

    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        raise ValueError(
            "Una de las dos clases quedo vacia. Revise dataset.normal_label "
            "en la configuracion contra los valores reales de la columna."
        )

    # ---- 2. Reparticionar segun el protocolo antifuga --------------------- #
    rng = np.random.default_rng(sp["seed"])
    benign_idx = rng.permutation(np.flatnonzero(y.to_numpy() == 0))
    attack_idx = np.flatnonzero(y.to_numpy() == 1)

    n_b = len(benign_idx)
    n_train = int(sp["train_frac"] * n_b)
    n_val = int(sp["val_frac"] * n_b)

    idx_train = benign_idx[:n_train]
    idx_val = benign_idx[n_train:n_train + n_val]
    idx_test_benign = benign_idx[n_train + n_val:]

    frac_atk = float(sp.get("attack_frac_in_test", 1.0))
    if frac_atk < 1.0:
        attack_idx = rng.permutation(attack_idx)[: int(frac_atk * len(attack_idx))]

    idx_test = np.concatenate([idx_test_benign, attack_idx])
    idx_test = rng.permutation(idx_test)

    # ---- 3. Escribir las particiones -------------------------------------- #
    out = splits_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    df_train = X.iloc[idx_train].reset_index(drop=True)
    df_val = X.iloc[idx_val].reset_index(drop=True)
    df_test = X.iloc[idx_test].reset_index(drop=True)
    df_test["y_true"] = y.iloc[idx_test].to_numpy()

    write_table(df_train, out / f"train.{fmt}")
    write_table(df_val, out / f"val.{fmt}")
    write_table(df_test, out / f"test.{fmt}")

    # ---- 4. Manifiesto de trazabilidad ------------------------------------ #
    manifest = {
        "dataset": cfg["paths"]["dataset_name"],
        "selector": cfg["paths"]["selector"],
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "seed": sp["seed"],
        "splits": {
            "train": {"rows": int(len(df_train)), "content": "solo normal, sin etiquetas"},
            "val": {"rows": int(len(df_val)), "content": "solo normal, sin etiquetas"},
            "test": {
                "rows": int(len(df_test)),
                "normal": int((df_test["y_true"] == 0).sum()),
                "attack": int((df_test["y_true"] == 1).sum()),
                "attack_ratio": round(float((df_test["y_true"] == 1).mean()), 4),
            },
        },
        "sha256": {
            name: sha256_of(out / f"{name}.{fmt}") for name in ("train", "val", "test")
        },
        "note": (
            "El bloque de prueba esta aislado: ningun script de optimizacion "
            "debe leerlo. Verifique el hash antes y despues del experimento."
        ),
    }
    save_json(manifest, out / "manifest.json")

    print("\nParticiones generadas en", out)
    for k, v in manifest["splits"].items():
        print(f"  {k:6s}: {v['rows']:,} filas")
    print(f"  ratio de ataque en test: {manifest['splits']['test']['attack_ratio']}")
    print(f"\nManifiesto: {out / 'manifest.json'}")
    save_json(manifest, outputs_dir(cfg) / "logs" / "splits_manifest.json")


if __name__ == "__main__":
    main()
