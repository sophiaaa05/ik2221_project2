"""
Plot accuracy, retrieval time, and inference time vs number of documents
in the RAG database. Produces two separate figures:
  - ndocs_accuracy.png
  - ndocs_latency.png

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
# Annotation — fixed direction per line so labels never collide
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_fixed(ax, xs, ys, fmt, color, fontsize=8.5, offset=12, direction=1):
    va = "bottom" if direction > 0 else "top"
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), xy=(x, y),
                    xytext=(0, direction * offset), textcoords="offset points",
                    ha="center", va=va, fontsize=fontsize, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy(summaries: list, output_dir: str) -> None:
    n   = [d["n_docs"]           for d in summaries]
    acc = [d["rag_accuracy_pct"] for d in summaries]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(n, acc, marker="o", linewidth=2.2, markersize=8,
            color=COLORS[0], label="Accuracy")
    ax.fill_between(n, acc, alpha=0.08, color=COLORS[0])
    _annotate_fixed(ax, n, acc, "{:.1f}%", color=COLORS[0], direction=+1)

    ax.set_xlabel("Number of documents in RAG database", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_xticks(n)
    ax.set_title("RAG Accuracy vs Number of Documents", fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="lower left")

    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir) / "ndocs_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Latency breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency(summaries: list, output_dir: str) -> None:
    n         = [d["n_docs"]                 for d in summaries]
    retrieval = [d["avg_retrieval_time_sec"]  for d in summaries]
    inference = [d["avg_inference_time_sec"]  for d in summaries]
    total     = [d["avg_latency_sec"]         for d in summaries]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(n, retrieval, marker="^", linewidth=2,   markersize=7,
            color=COLORS[1], linestyle="--",  label="Retrieval time")
    ax.plot(n, inference, marker="s", linewidth=2,   markersize=7,
            color=COLORS[2], linestyle="-.",  label="Inference time")
    ax.plot(n, total,     marker="o", linewidth=2.2, markersize=8,
            color=COLORS[3],                  label="Total avg latency")

    # retrieval near zero → label below; inference below total; total above
    _annotate_fixed(ax, n, retrieval, "{:.4f}s", color=COLORS[1],
                    fontsize=8, offset=10, direction=-1)
    _annotate_fixed(ax, n, inference, "{:.3f}s",  color=COLORS[2],
                    fontsize=8, offset=10, direction=-1)
    _annotate_fixed(ax, n, total,     "{:.3f}s",  color=COLORS[3],
                    fontsize=8, offset=12, direction=+1)

    ax.set_xlabel("Number of documents in RAG database", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_xticks(n)
    ax.set_title("Latency vs Number of Documents", fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="upper left")

    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir) / "ndocs_latency.png"
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

    print("\nGenerating figures...")
    plot_accuracy(summaries, args.output_dir)
    plot_latency(summaries,  args.output_dir)
    print("All done!")


if __name__ == "__main__":
    main()