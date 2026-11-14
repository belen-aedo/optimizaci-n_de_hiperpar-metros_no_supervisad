"""
prepare_splits.py — Etapa 0 (bloque 2 del diagrama metodologico).

Tres bloques, como en la propuesta aprobada:

    train : SOLO trafico normal, sin etiquetas
    val   : SOLO trafico normal, sin etiquetas  (calibracion del umbral + objetivo)
    test  : normal + ataque, etiquetado, aislado. Se SUBDIVIDE (estratificado,
            semilla fija) en dos partes disjuntas:
              test_oracle : la usa SOLO el escenario oraculo para maximizar MCC
              test_final  : se abre una sola vez (evaluate.py); igual para los
                            tres escenarios

El oraculo necesita etiquetas para optimizar; si usara los mismos datos con que
luego se evalua, su cota superior quedaria inflada. De ahi la subdivision.

Opcional (split.refit_scaler): reescala todas las columnas con media y
desviacion del TRAIN NORMAL. Las salidas de preprocessor_v2 se ajustaron con
trafico mixto (incluye ataques); esto evita que esa informacion llegue al
detector. NO corrige la seleccion de variables si uso etiquetas (L1, ANOVA,
informacion mutua, chi2): ver README_CAMBIOS.md.

USO
    python src/prepare_splits.py --input-format csv --format csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (load_config, outputs_dir, read_table, save_json,
                    selector_paths, set_seeds, sha256_of, splits_dir, write_table)


def build_target(df: pd.DataFrame, cfg) -> pd.Series:
    """1 = anomalia, 0 = normal."""
    label_col = cfg["dataset"]["label_col"]
    normal = str(cfg["dataset"]["normal_label"]).lower()
    if label_col not in df.columns:
        raise KeyError(f"La columna de etiqueta '{label_col}' no esta en los datos. "
                       f"Columnas: {list(df.columns)[:20]} ...")
    raw = df[label_col].astype(str).str.strip().str.lower()
    return (raw != normal).astype(int)


def feature_matrix(df: pd.DataFrame, cfg) -> pd.DataFrame:
    drop = [cfg["dataset"]["label_col"], *cfg["dataset"].get("metadata_cols", [])]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32)


def stratified_halves(rng, y: np.ndarray, idx: np.ndarray, frac_first: float):
    """Divide idx en (primera, segunda) conservando la proporcion de clases."""
    first, second = [], []
    for cls in (0, 1):
        sub = rng.permutation(idx[y[idx] == cls])
        k = int(round(frac_first * len(sub)))
        first.append(sub[:k])
        second.append(sub[k:])
    return rng.permutation(np.concatenate(first)), rng.permutation(np.concatenate(second))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--input-format", default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp = cfg["split"]
    set_seeds(sp["seed"])

    train_path, test_path = selector_paths(cfg, args.input_format)
    df = pd.concat([read_table(train_path), read_table(test_path)], ignore_index=True)
    y = build_target(df, cfg).to_numpy()
    X = feature_matrix(df, cfg)
    print(f"{len(df):,} filas | {X.shape[1]} caracteristicas | "
          f"normal {(y == 0).sum():,} | ataque {(y == 1).sum():,}")
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        raise ValueError("Una de las clases quedo vacia: revise dataset.normal_label.")

    rng = np.random.default_rng(sp["seed"])
    benign = rng.permutation(np.flatnonzero(y == 0))
    attack = np.flatnonzero(y == 1)
    n_b = len(benign)
    n_tr, n_va = int(sp["train_frac"] * n_b), int(sp["val_frac"] * n_b)
    idx_train, idx_val = benign[:n_tr], benign[n_tr:n_tr + n_va]
    idx_test_b = benign[n_tr + n_va:]

    if sp["attack_frac_in_test"] < 1.0:
        attack = rng.permutation(attack)[: int(sp["attack_frac_in_test"] * len(attack))]
    idx_test = np.concatenate([idx_test_b, attack])
    idx_orc, idx_fin = stratified_halves(rng, y, idx_test, sp["oracle_frac_of_test"])

    # ---- escalado ajustado SOLO con train normal ----------------------------
    Xv = X.to_numpy(dtype=np.float64)
    scaler = None
    if sp["refit_scaler"]:
        mu = Xv[idx_train].mean(axis=0)
        sd = Xv[idx_train].std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xv = (Xv - mu) / sd
        scaler = {"mean": mu.tolist(), "std": sd.tolist()}
    Xdf = pd.DataFrame(Xv.astype(np.float32), columns=X.columns)

    out = splits_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    fmt = args.format
    for name, idx in (("train", idx_train), ("val", idx_val)):
        write_table(Xdf.iloc[idx].reset_index(drop=True), out / f"{name}.{fmt}")
    for name, idx in (("test_oracle", idx_orc), ("test_final", idx_fin)):
        d = Xdf.iloc[idx].reset_index(drop=True)
        d["y_true"] = y[idx]
        write_table(d, out / f"{name}.{fmt}")
    if scaler:
        save_json(scaler, out / "scaler.json")

    def block(idx):
        return {"rows": int(len(idx)), "normal": int((y[idx] == 0).sum()),
                "attack": int((y[idx] == 1).sum()),
                "attack_ratio": round(float((y[idx] == 1).mean()), 4)}

    manifest = {
        "dataset": cfg["paths"]["dataset_name"], "selector": cfg["paths"]["selector"],
        "n_features": int(X.shape[1]), "feature_names": list(X.columns),
        "seed": sp["seed"], "scaler_refit_on_train_normal": bool(scaler),
        "splits": {
            "train": {"rows": int(len(idx_train)), "content": "solo normal, sin etiquetas"},
            "val": {"rows": int(len(idx_val)), "content": "solo normal, sin etiquetas"},
            "test_oracle": {**block(idx_orc), "use": "solo escenario oraculo"},
            "test_final": {**block(idx_fin), "use": "solo evaluate.py, una vez"},
        },
        "sha256": {n: sha256_of(out / f"{n}.{fmt}")
                   for n in ("train", "val", "test_oracle", "test_final")},
        "note": ("test_final esta aislado: ningun script de optimizacion debe leerlo. "
                 "Verifique el hash antes y despues del experimento."),
    }
    save_json(manifest, out / "manifest.json")
    save_json(manifest, outputs_dir(cfg) / "logs" / "splits_manifest.json")

    print("\nParticiones en", out)
    for k, v in manifest["splits"].items():
        extra = f" | ataque {v['attack_ratio']:.1%}" if "attack_ratio" in v else ""
        print(f"  {k:12s}: {v['rows']:,} filas{extra}")
    ratio = manifest["splits"]["test_final"]["attack_ratio"]
    if ratio > 0.5:
        print(f"\n[!] AVISO: el {ratio:.0%} de test_final es ataque. Con esa prevalencia "
              "la Precision es enganosamente alta y el MCC no representa un despliegue "
              "real. evaluate.py reporta tambien MCC ajustado a prevalencia "
              f"{sp['report_prevalences']}; ademas conviene usar mas trafico normal.")
    if manifest["splits"]["test_oracle"]["normal"] < 300:
        print("[!] AVISO: pocos normales en test_oracle: el FPR estimado sera ruidoso.")


if __name__ == "__main__":
    main()
