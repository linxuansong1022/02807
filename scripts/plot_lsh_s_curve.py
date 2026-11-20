"""Plot MinHash S curves against exact top-N Jaccard scores and optional scale results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

if __package__:
    from . import benchmark_retrieval, retrieval
else:
    import benchmark_retrieval
    import retrieval


def s_curve_figure(p10: float, p90: float, target_probability: float | None):
    import matplotlib.pyplot as plt

    similarities = np.linspace(0, 1, 501)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5), sharey=True, layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for ax, hashes in zip(axes, benchmark_retrieval.HASH_COUNTS, strict=True):
        ax.axvspan(p10, p90, color="#b2c5d7", alpha=.38)
        ax.axvline(p10, color="#344663", linestyle=":", linewidth=1.2)
        for i, (bands, rows) in enumerate(benchmark_retrieval.band_grid(hashes)):
            ax.plot(similarities, retrieval.candidate_probability(similarities, bands, rows),
                    label=f"{bands} × {rows}", color=colors[i],
                    linestyle="-" if i < 6 else "--", linewidth=1.5)
        if target_probability is not None:
            ax.axhline(target_probability, color="#39414a", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Jaccard similarity", title=f"{hashes} hashes")
        ax.grid(axis="y", alpha=.17)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False,
                  fontsize=8, ncol=3, title="Bands × rows")
    axes[0].set_ylabel("Candidate probability")
    caption = f"Positive exact neighbors: P10–P90 {p10:.3f}–{p90:.3f} (shaded)"
    if target_probability is not None:
        caption += f"   |   Example target {target_probability:.0%} (dashed)"
    fig.suptitle(f"MinHash banding and exact neighbors\n{caption}")
    return fig


def scaling_figure(rows, top_n: int):
    import matplotlib.pyplot as plt

    series = [
        ("Sparse exact", "", "#212932", "o"),
        ("32 × 4", "32:4", "#446e9b", "s"),
        ("64 × 2", "64:2", "#b45a43", "^"),
        ("128 × 2", "128:2", "#3d8367", "D"),
        ("128 × 1", "128:1", "#795f9c", "v"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2), layout="constrained")
    for label, key, color, marker in series:
        selected = [row for row in rows if (
            row["method"] == "sparse_exact" if not key else f"{row['bands']}:{row['rows']}" == key
        )]
        selected.sort(key=lambda row: int(row["index_tracks"]))
        sizes = [int(row["index_tracks"]) for row in selected]
        for ax, metric in zip(axes, ("query_p50_ms", "neighbor_recall"), strict=True):
            ax.plot(sizes, [float(row[metric]) for row in selected], color=color, marker=marker,
                    label=label, linewidth=1.55, markersize=4,
                    linestyle="--" if not key else "-", zorder=3 if not key else 2)
    axes[0].set(yscale="log", ylabel="Median query latency (ms)")
    axes[1].set(ylim=(0, 1.04), ylabel=f"Positive-neighbor Recall@{top_n}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(series), frameon=False, fontsize=9)
    sizes = sorted({int(row["index_tracks"]) for row in rows})
    for ax in axes:
        ax.set_xlabel("Indexed songs")
        ax.set_xticks(sizes, [f"{size / 1000:.1f}k" for size in sizes])
        ax.grid(axis="y", alpha=.17)
    fig.suptitle("Retrieval latency and quality across catalog sizes")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbor-jaccard", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="S-curve PNG or PDF path")
    parser.add_argument("--target-probability", type=float)
    parser.add_argument("--benchmark", type=Path, help="Also render the scale figure from this CSV")
    args = parser.parse_args()
    if args.target_probability is not None and not 0 <= args.target_probability <= 1:
        parser.error("--target-probability must be between 0 and 1")
    output = args.output or args.neighbor_jaccard.with_name("lsh_s_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    scores = np.load(args.neighbor_jaccard, allow_pickle=False)
    observed = scores[np.isfinite(scores)]
    if np.any((observed < 0) | (observed > 1)) or not np.any(observed > 0):
        raise ValueError("expected at least one positive Jaccard score and all finite scores in [0, 1]")
    p10, p90 = np.percentile(observed[observed > 0], (10, 90))
    with matplotlib.rc_context({
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titleweight": "bold", "pdf.fonttype": 42,
    }):
        s_curve_figure(p10, p90, args.target_probability).savefig(output, dpi=300)
        if args.benchmark:
            with args.benchmark.open(newline="") as file:
                figure = scaling_figure(list(csv.DictReader(file)), scores.shape[-1])
            figure.savefig(output.with_name("retrieval_scaling.png"), dpi=300)


if __name__ == "__main__":
    main()
