"""
Plot accuracy, retrieval time, and inference time vs number of documents
in the RAG database.

Usage:
    python plot_ndocs.py --results-dir results --output-dir plots
"""

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})
COLORS = plt.cm.tab10.colors


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_ndocs_summaries(results_dir: str) -> list:
    summaries = []
    for path in Path(results_dir).glob("*_summary.json"):
        with open(path) as f:
            data = json.load(f)
        label = data.get("label", path.stem.replace("_summary", ""))
        # Accept labels like ndocs_4, ndocs_14, docs_7, n_docs_10 …
        m = re.search(r"ndocs?[_\-]?(\d+)", label, re.IGNORECASE)
        if not m:
            continue
        data["label"]  = label
        data["n_docs"] = int(m.group(1))
        summaries.append(data)

    if not summaries:
        raise SystemExit(
            "No ndocs_* summary files found. "
            "Make sure RESULTS_LABEL=ndocs_<N> when running experiments."
        )

    summaries.sort(key=lambda d: d["n_docs"])
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper — flips label below the point when neighbours overlap
# ─────────────────────────────────────────────────────────────────────────────

def _annotate(ax, xs, ys, fmt, color, fontsize=8.5, base_offset=10):
    fig = ax.get_figure()
    fig.canvas.draw()
    tr = ax.transData

    disp_y    = [tr.transform((x, y))[1] for x, y in zip(xs, ys)]
    prev_dy   = None
    prev_sign = +1

    for x, y, dy in zip(xs, ys, disp_y):
        if prev_dy is not None and abs(dy - prev_dy) < 18:
            sign = -prev_sign
        else:
            sign = +1
        ax.annotate(fmt.format(y), xy=(x, y),
                    xytext=(0, sign * base_offset), textcoords="offset points",
                    ha="center", va="bottom" if sign > 0 else "top",
                    fontsize=fontsize, color=color)
        prev_dy   = dy
        prev_sign = sign


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_ndocs(summaries: list, output_dir: str) -> None:
    n        = [d["n_docs"]                for d in summaries]
    acc      = [d["rag_accuracy_pct"]      for d in summaries]
    retrieval= [d["avg_retrieval_time_sec"] for d in summaries]
    inference= [d["avg_inference_time_sec"] for d in summaries]
    total    = [d["avg_latency_sec"]        for d in summaries]

    fig, (ax_acc, ax_lat) = plt.subplots(
        2, 1, figsize=(8, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    # ── Top panel: Accuracy ──────────────────────────────────────────────────
    ax_acc.plot(n, acc, marker="o", linewidth=2.2, markersize=8,
                color=COLORS[0], label="Accuracy")
    ax_acc.fill_between(n, acc, alpha=0.08, color=COLORS[0])

    _annotate(ax_acc, n, acc, "{:.1f}%", color=COLORS[0])

    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_ylim(0, 115)
    ax_acc.set_title("RAG Performance vs Number of Documents in the Database",
                     fontsize=13, pad=14)
    ax_acc.legend(fontsize=10, loc="lower left")
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    # ── Bottom panel: Latencies ──────────────────────────────────────────────
    ax_lat.plot(n, retrieval, marker="^", linewidth=2,   markersize=7,
                color=COLORS[1], linestyle="--",  label="Retrieval time")
    ax_lat.plot(n, inference, marker="s", linewidth=2,   markersize=7,
                color=COLORS[2], linestyle="-.",  label="Inference time")
    ax_lat.plot(n, total,     marker="o", linewidth=2.2, markersize=8,
                color=COLORS[3],                  label="Total avg latency")

    _annotate(ax_lat, n, retrieval, "{:.4f}s", color=COLORS[1], fontsize=8)
    _annotate(ax_lat, n, inference, "{:.3f}s",  color=COLORS[2], fontsize=8)
    _annotate(ax_lat, n, total,     "{:.3f}s",  color=COLORS[3], fontsize=8)

    ax_lat.set_xlabel("Number of documents in RAG database", fontsize=11)
    ax_lat.set_ylabel("Time (s)", fontsize=11)
    ax_lat.set_xticks(n)
    ax_lat.legend(fontsize=10, loc="upper left")
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir) / "ndocs_accuracy_latency.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="plots")
    args = parser.parse_args()

    summaries = load_ndocs_summaries(args.results_dir)
    print(f"Loaded {len(summaries)} ndocs experiment(s):")
    for d in summaries:
        print(f"  n={d['n_docs']:3d}  acc={d['rag_accuracy_pct']:.1f}%  "
              f"retrieval={d['avg_retrieval_time_sec']:.4f}s  "
              f"inference={d['avg_inference_time_sec']:.3f}s  "
              f"total={d['avg_latency_sec']:.3f}s")

    plot_ndocs(summaries, args.output_dir)


if __name__ == "__main__":
    main()