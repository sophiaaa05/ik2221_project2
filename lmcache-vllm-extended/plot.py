"""
IK2221 Task 1 - Plotting Script
Reads both old single-trial and new multi-trial averaged summary files.
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from glob import glob

CACHE_SIZES = ["0GB",  "1GB", "2GB", "4GB",  "8GB"]

SIZE_LABELS = {
    "0GB": "0GB", "0p5GB": "0.5GB", "1GB": "1GB", "2GB": "2GB",
    "3GB": "3GB", "4GB": "4GB",     "6GB": "6GB", "8GB": "8GB"
}

RUN_TYPES = ["single", "repeat", "diverse", "diverse_more_contexts", "diverse_more_repeat"]

COLORS = {
    "single":                "#2196F3",
    "repeat":                "#4CAF50",
    "diverse":               "#FF5722",
    "diverse_more_contexts": "#9C27B0",
    "diverse_more_repeat":   "#FF9800",
}

RUN_LABELS = {
    "single":                "Single (low diversity)",
    "repeat":                "Repeat (cache hit)",
    "diverse":               "Diverse (different order)",
    "diverse_more_contexts": "Diverse (more contexts)",
    "diverse_more_repeat":   "Diverse more (repeated)",
}

# ── Load summaries — reads BOTH old and new format ───────────────────────────

def load_summaries(results_dir):
    """
    Priority: averaged summary (_avg_summary.json) > single trial summary.
    This way old results still show up if no averaged version exists.
    """
    summaries = {}

    # Pattern for new averaged files: cache_4GB_single_avg_summary.json
    avg_pattern = re.compile(
        r'cache_(\w+GB)_(single|repeat|diverse|diverse_more_contexts|diverse_more_repeat)_avg_summary\.json'
    )
    # Pattern for old single-trial files: cache_4GB_single_20260511_..._summary.json
    old_pattern = re.compile(
        r'cache_(\w+GB)_(single|repeat|diverse|diverse_more_contexts|diverse_more_repeat)_\d{8}_\d{6}_summary\.json'
    )

    # Load old format first (lower priority)
    for fpath in sorted(glob(os.path.join(results_dir, "*_summary.json"))):
        fname = os.path.basename(fpath)
        m = old_pattern.match(fname)
        if not m:
            continue
        size, rtype = m.groups()
        with open(fpath) as f:
            data = json.load(f)
        key = (size, rtype)
        if key not in summaries:  # don't overwrite if already loaded
            summaries[key] = data
            print(f"  Loaded old summary:  {size:8s} {rtype}")

    # Load new averaged format (higher priority — overwrites old)
    for fpath in sorted(glob(os.path.join(results_dir, "*_avg_summary.json"))):
        fname = os.path.basename(fpath)
        m = avg_pattern.match(fname)
        if not m:
            continue
        size, rtype = m.groups()
        with open(fpath) as f:
            data = json.load(f)
        summaries[(size, rtype)] = data  # always overwrite with averaged
        print(f"  Loaded avg summary:  {size:8s} {rtype}")

    return summaries


# ── Load per-request results — reads BOTH old and new format ─────────────────

def load_all_results(results_dir):
    """
    Combines trial results files AND old single-run results files.
    """
    all_data = {}

    # New format: cache_4GB_single_trial1_20260511_results.json
    trial_pattern = re.compile(
        r'cache_(\w+GB)_(single|repeat|diverse|diverse_more_contexts|diverse_more_repeat)_trial\d+_.*_results\.json'
    )
    # Old format: cache_4GB_single_20260511_..._results.json
    old_pattern = re.compile(
        r'cache_(\w+GB)_(single|repeat|diverse|diverse_more_contexts|diverse_more_repeat)_\d{8}_\d{6}_results\.json'
    )

    for fpath in sorted(glob(os.path.join(results_dir, "*_results.json"))):
        fname = os.path.basename(fpath)
        m = trial_pattern.match(fname) or old_pattern.match(fname)
        if not m:
            continue
        size, rtype = m.groups()
        with open(fpath) as f:
            data = json.load(f)
        key = (size, rtype)
        if key not in all_data:
            all_data[key] = []
        all_data[key].extend(data)
        print(f"  Loaded results:      {size:8s} {rtype} ({len(data)} records)")

    return all_data


# ── GRAPH 1: Cache size vs Avg Latency ───────────────────────────────────────

def plot_cache_vs_latency(summaries, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for rtype in RUN_TYPES:
        latencies   = []
        xlabels     = []
        for size in CACHE_SIZES:
            if (size, rtype) in summaries:
                latencies.append(summaries[(size, rtype)]["avg_latency_sec"])
                xlabels.append(SIZE_LABELS[size])
        if latencies:
            ax.plot(xlabels, latencies, marker="o",
                    label=RUN_LABELS[rtype], color=COLORS[rtype], linewidth=2)

    ax.set_xlabel("Local Cache Size", fontsize=12)
    ax.set_ylabel("Average Latency (s)", fontsize=12)
    ax.set_title("Cache Size vs Average Latency", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "cache_vs_latency.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── GRAPH 2: Cache size vs Throughput ────────────────────────────────────────

def plot_cache_vs_throughput(summaries, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    for rtype in RUN_TYPES:
        throughputs = []
        xlabels     = []
        for size in CACHE_SIZES:
            if (size, rtype) in summaries:
                throughputs.append(summaries[(size, rtype)]["throughput_req_per_sec"])
                xlabels.append(SIZE_LABELS[size])
        if throughputs:
            ax.plot(xlabels, throughputs, marker="o",
                    label=RUN_LABELS[rtype], color=COLORS[rtype], linewidth=2)

    ax.set_xlabel("Local Cache Size", fontsize=12)
    ax.set_ylabel("Throughput (req/s)", fontsize=12)
    ax.set_title("Cache Size vs Throughput", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "cache_vs_throughput.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── GRAPH 3: Sequence Length vs Latency ──────────────────────────────────────

def plot_seqlen_vs_latency(all_results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes_present = [s for s in CACHE_SIZES if (s, "single") in all_results]
    colors = cm.viridis(np.linspace(0, 1, len(sizes_present)))

    for i, size in enumerate(sizes_present):
        data = all_results[(size, "single")]
        xs = [r["prompt_len"] for r in data if r["latency"] is not None]
        ys = [r["latency"]    for r in data if r["latency"] is not None]
        ax.scatter(xs, ys, alpha=0.4, s=20, color=colors[i],
                   label=f"{SIZE_LABELS[size]} cache")
        if len(xs) > 1:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            xs_s = sorted(xs)
            ax.plot(xs_s, [p(x) for x in xs_s],
                    color=colors[i], linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Prompt Length (characters)", fontsize=12)
    ax.set_ylabel("Latency (s)", fontsize=12)
    ax.set_title("Sequence Length vs Latency (Single Pass)", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "seqlen_vs_latency.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── GRAPH 4: Cache Hit Effect ─────────────────────────────────────────────────

def plot_cache_hit_effect(all_results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    first_avgs  = []
    repeat_avgs = []
    improvements = []
    xlabels     = []

    for size in CACHE_SIZES:
        if (size, "repeat") not in all_results:
            continue
        data = all_results[(size, "repeat")]
        seen = {}
        first_lats, repeat_lats = [], []
        for r in data:
            if r["latency"] is None:
                continue
            uid = (r["context_id"], r["question"])
            if uid not in seen:
                seen[uid] = True
                first_lats.append(r["latency"])
            else:
                repeat_lats.append(r["latency"])
        if first_lats and repeat_lats:
            avg1 = np.mean(first_lats)
            avg2 = np.mean(repeat_lats)
            first_avgs.append(avg1)
            repeat_avgs.append(avg2)
            improvements.append((avg1 - avg2) / avg1 * 100)
            xlabels.append(SIZE_LABELS[size])

    if xlabels:
        x = np.arange(len(xlabels))
        width = 0.35
        bars1 = ax.bar(x - width/2, first_avgs,  width,
                       label="First occurrence",    color="#2196F3")
        bars2 = ax.bar(x + width/2, repeat_avgs, width,
                       label="Repeated occurrence", color="#4CAF50")

        # Add % improvement labels on top
        for xi, imp in zip(x, improvements):
            ax.text(xi, max(first_avgs[list(x).index(xi)],
                            repeat_avgs[list(x).index(xi)]) + 0.005,
                    f"{imp:.1f}%", ha="center", fontsize=9, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Local Cache Size", fontsize=12)
        ax.set_ylabel("Average Latency (s)", fontsize=12)
        ax.set_title("Cache Hit Effect: First vs Repeated Requests", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(output_dir, "cache_hit_effect.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── GRAPH 5: Diversity Effect ─────────────────────────────────────────────────

def plot_diversity_effect(summaries, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    lines = [
        ("single",                "Low diversity (single order)",    "#2196F3"),
        ("diverse",               "High diversity (different order)", "#FF5722"),
        ("diverse_more_contexts", "High diversity (more contexts)",   "#9C27B0"),
    ]

    for rtype, label, color in lines:
        latencies = []
        xlabels   = []
        for size in CACHE_SIZES:
            if (size, rtype) in summaries:
                latencies.append(summaries[(size, rtype)]["avg_latency_sec"])
                xlabels.append(SIZE_LABELS[size])
        if latencies:
            ax.plot(xlabels, latencies, marker="o",
                    label=label, color=color, linewidth=2)

    ax.set_xlabel("Local Cache Size", fontsize=12)
    ax.set_ylabel("Average Latency (s)", fontsize=12)
    ax.set_title("Effect of Request Diversity on Latency", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "diversity_effect.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

def plot_seqlen_vs_latency_expanded(all_results, output_dir):
    """Use diverse_more_contexts data which has wider prompt length range."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes_present = [s for s in CACHE_SIZES if (s, "diverse_more_contexts") in all_results]
    colors = cm.viridis(np.linspace(0, 1, len(sizes_present)))

    for i, size in enumerate(sizes_present):
        data = all_results[(size, "diverse_more_contexts")]
        xs = [r["prompt_len"] for r in data if r["latency"] is not None]
        ys = [r["latency"]    for r in data if r["latency"] is not None]
        ax.scatter(xs, ys, alpha=0.4, s=20, color=colors[i],
                   label=f"{SIZE_LABELS[size]} cache")
        if len(xs) > 1:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            xs_s = sorted(xs)
            ax.plot(xs_s, [p(x) for x in xs_s],
                    color=colors[i], linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Prompt Length (characters)", fontsize=12)
    ax.set_ylabel("Latency (s)", fontsize=12)
    ax.set_title("Sequence Length vs Latency (Expanded Contexts)", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "seqlen_vs_latency_expanded.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IK2221 Task 1 - Plot Results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading summaries (old + averaged)...")
    summaries = load_summaries(args.results_dir)
    print(f"  Total: {len(summaries)} summaries loaded.")

    print("\nLoading per-request results (old + trial)...")
    all_results = load_all_results(args.results_dir)
    print(f"  Total: {len(all_results)} (cache_size, run_type) combinations.")

    print("\nGenerating plots...")
    plot_cache_vs_latency(summaries, args.output_dir)
    plot_cache_vs_throughput(summaries, args.output_dir)
    plot_seqlen_vs_latency(all_results, args.output_dir)
    plot_cache_hit_effect(all_results, args.output_dir)
    plot_diversity_effect(summaries, args.output_dir)
    plot_seqlen_vs_latency_expanded(all_results, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()