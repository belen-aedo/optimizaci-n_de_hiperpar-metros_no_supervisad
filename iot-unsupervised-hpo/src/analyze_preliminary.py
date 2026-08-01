"""
analyze_benchmark.py
--------------------
Generador de evidencia (Etapa-1) para el flujo experimental.

Lee el benchmark de selección de características producido por `run_feature_selection.py`
(outputs/ml_benchmark_results.csv) y genera:

    outputs/tables/table_selector_summary.tex   (LaTeX, estilo IEEE)
    outputs/tables/table_full_benchmark.tex     (LaTeX, estilo IEEE)
    outputs/figures/fig_selector_benchmark.png  (figura para el artículo)
    outputs/selected_feature_space.json         (config consumida por la Etapa-2)

Uso:
    python src/analyze_benchmark.py --input outputs/ml_benchmark_results.csv \
                                    --outdir outputs
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["Accuracy", "F1-macro", "MCC", "Balanced-Acc", "ROC-AUC", "Train_time_s"]
PALETTE = {"DecisionTree": "#1b6ca8", "RandomForest": "#2a9d8f", "LogisticRegression": "#e07a5f"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in METRICS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def selector_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("Selector")
        .agg(
            N_features=("N_features", "first"),
            MCC_mean=("MCC", "mean"),
            MCC_best=("MCC", "max"),
            F1_mean=("F1-macro", "mean"),
            BACC_mean=("Balanced-Acc", "mean"),
            AUC_mean=("ROC-AUC", "mean"),
            Time_mean=("Train_time_s", "mean"),
        )
        .sort_values("MCC_best", ascending=False)
        .round(4)
    )
    return agg.reset_index()


def write_selector_table(summary: pd.DataFrame, path: str) -> None:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"{r['Selector'].replace('_', chr(92)+'_')} & {int(r['N_features'])} & "
            f"{r['MCC_mean']:.3f} & {r['MCC_best']:.3f} & {r['F1_mean']:.3f} & "
            f"{r['BACC_mean']:.3f} & {r['AUC_mean']:.3f} & {r['Time_mean']:.2f} \\\\"
        )
    body = "\n".join(rows)
    tex = f"""\\begin{{table}}[t]
\\caption{{Stage-1 feature-selection benchmark on ToN-IoT (real execution).
Values are averaged over the three supervised reference classifiers.}}
\\label{{tab:selector}}
\\centering
\\footnotesize
\\begin{{tabular}}{{lccccccc}}
\\hline
\\textbf{{Selector}} & \\textbf{{$|F|$}} & \\textbf{{MCC}} & \\textbf{{MCC}} &
\\textbf{{F1}} & \\textbf{{BACC}} & \\textbf{{AUC}} & \\textbf{{t (s)}} \\\\
 & & \\textbf{{avg}} & \\textbf{{best}} & \\textbf{{avg}} & \\textbf{{avg}} &
\\textbf{{avg}} & \\textbf{{avg}} \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


def write_full_table(df: pd.DataFrame, path: str) -> None:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['Selector'].replace('_', chr(92)+'_')} & {int(r['N_features'])} & "
            f"{r['Model']} & {r['MCC']:.4f} & {r['F1-macro']:.4f} & "
            f"{r['Balanced-Acc']:.4f} & {r['ROC-AUC']:.4f} & {r['Train_time_s']:.2f} \\\\"
        )
    body = "\n".join(rows)
    tex = f"""\\begin{{table}}[t]
\\caption{{Complete Stage-1 benchmark grid (15 real runs, ToN-IoT).}}
\\label{{tab:fullbench}}
\\centering
\\scriptsize
\\begin{{tabular}}{{llcccccc}}
\\hline
\\textbf{{Selector}} & \\textbf{{$|F|$}} & \\textbf{{Model}} & \\textbf{{MCC}} &
\\textbf{{F1}} & \\textbf{{BACC}} & \\textbf{{AUC}} & \\textbf{{t (s)}} \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


LABELS = {
    "es": {
        "a": "(a) Calidad de deteccion por estrategia de reduccion",
        "b": "(b) Compromiso entre calidad y costo computacional",
        "x": "Tiempo de entrenamiento (s)",
    },
    "en": {
        "a": "(a) Detection quality per feature selector",
        "b": "(b) Quality vs. computational cost",
        "x": "Training time (s)",
    },
}


def make_figure(df: pd.DataFrame, path: str, lang: str = "es") -> None:
    txt = LABELS[lang]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    order = (
        df.groupby("Selector")["MCC"].max().sort_values(ascending=False).index.tolist()
    )
    models = ["DecisionTree", "RandomForest", "LogisticRegression"]
    width = 0.26
    ax = axes[0]
    for i, m in enumerate(models):
        sub = df[df["Model"] == m].set_index("Selector").loc[order]
        xs = [j + (i - 1) * width for j in range(len(order))]
        ax.bar(xs, sub["MCC"], width=width, label=m, color=PALETTE[m], edgecolor="white")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
    ax.set_ylabel("MCC")
    ax.set_ylim(0.6, 1.0)
    ax.set_title(txt["a"], fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[1]
    for m in models:
        sub = df[df["Model"] == m]
        ax.scatter(
            sub["Train_time_s"], sub["MCC"], s=55, label=m,
            color=PALETTE[m], edgecolor="white", zorder=3,
        )
    for _, r in df.iterrows():
        ax.annotate(
            r["Selector"].split("_")[0], (r["Train_time_s"], r["MCC"]),
            fontsize=6, xytext=(3, 3), textcoords="offset points", alpha=0.75,
        )
    ax.set_xlabel(txt["x"])
    ax.set_ylabel("MCC")
    ax.set_title(txt["b"], fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/ml_benchmark_results.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--lang", default="es", choices=["es", "en"],
                    help="Idioma de los rotulos de la figura")
    args = ap.parse_args()

    df = load(args.input)
    os.makedirs(os.path.join(args.outdir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)

    summary = selector_summary(df)
    write_selector_table(summary, os.path.join(args.outdir, "tables", "table_selector_summary.tex"))
    write_full_table(df, os.path.join(args.outdir, "tables", "table_full_benchmark.tex"))
    make_figure(df, os.path.join(args.outdir, "figures", "fig_selector_benchmark.png"), lang=args.lang)

    best = summary.iloc[0]
    config = {
        "dataset": "ToN-IoT",
        "selected_selector": str(best["Selector"]),
        "n_features": int(best["N_features"]),
        "supervised_reference_mcc": float(best["MCC_best"]),
        "note": "Feature space frozen for the Stage-2 unsupervised optimisation loop.",
    }
    with open(os.path.join(args.outdir, "selected_feature_space.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    print(summary.to_string(index=False))
    print("\nSelected feature space ->", config["selected_selector"], f"({config['n_features']} features)")


if __name__ == "__main__":
    main()
