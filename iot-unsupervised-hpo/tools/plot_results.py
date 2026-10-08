from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DETECTOR_LABELS = {
	"autoencoder": "Autoencoder",
	"iforest": "Isolation Forest",
	"lof": "LOF",
}
METRIC_LABELS = {
	"por_defecto": "Por defecto",
	"davies_bouldin": "Davies-Bouldin",
	"reconstruction": "Reconstruction",
	"silhouette": "Silhouette",
	"stability": "Stability",
}
COLORS = {"Autoencoder": "#377eb8", "Isolation Forest": "#ff7f00", "LOF": "#4daf4a"}


def _prepare(df: pd.DataFrame, value: str) -> pd.DataFrame:
	required = {"detector", "metric", value}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"Faltan columnas en los resultados: {sorted(missing)}")
	result = df.copy()
	result["Detector"] = result["detector"].map(DETECTOR_LABELS).fillna(result["detector"])
	result["Criterion"] = result["metric"].map(METRIC_LABELS).fillna(result["metric"])
	return result.groupby(["metric", "Criterion", "Detector"], as_index=False)[value].mean()


def _style_axis(ax) -> None:
	ax.grid(axis="y", alpha=0.25)
	ax.set_axisbelow(True)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)


def make_figure(df: pd.DataFrame, value: str, ylabel: str, title: str, path: Path) -> None:
	data = _prepare(df, value)
	metric_order = [metric for metric in METRIC_LABELS if metric in data["metric"].unique()]
	detectors = list(DETECTOR_LABELS.values())
	positions = range(len(metric_order))
	width = 0.24

	fig, ax = plt.subplots(figsize=(11, 6))
	for index, detector in enumerate(detectors):
		values = []
		for metric in metric_order:
			match = data[(data["metric"] == metric) & (data["Detector"] == detector)]
			values.append(float(match[value].iloc[0]) if not match.empty else 0.0)
		offsets = [position + (index - 1) * width for position in positions]
		ax.bar(offsets, values, width=width, label=detector, color=COLORS[detector])

	ax.set_title(title)
	ax.set_xlabel("Métrica")
	ax.set_ylabel(ylabel)
	ax.set_xticks(list(positions))
	ax.set_xticklabels([METRIC_LABELS[metric] for metric in metric_order], rotation=45, ha="right")
	ax.legend(title="Detector")
	_style_axis(ax)
	fig.tight_layout()
	fig.savefig(path, dpi=220)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--results", default="outputs/results/final_evaluation.csv")
	parser.add_argument("--hardware", default="outputs/results/edge_benchmark.csv")
	parser.add_argument("--outdir", default="outputs/figures")
	args = parser.parse_args()

	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)
	performance = pd.read_csv(args.results)
	hardware = pd.read_csv(args.hardware)
	make_figure(
		performance,
		"MCC",
		"MCC medio",
		"Desempeño de detección: MCC medio por detector y métrica",
		outdir / "performance_mcc.png",
	)
	make_figure(
		hardware,
		"latency_median_ms",
		"Latencia mediana (ms)",
		"Latencia mediana por detector y métrica (Raspberry Pi 5)",
		outdir / "hardware_latency.png",
	)
	print(f"Figuras guardadas en {outdir}")


if __name__ == "__main__":
	main()
