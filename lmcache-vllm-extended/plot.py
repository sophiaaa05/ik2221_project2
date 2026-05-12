"""
=======================================================
Plots cache comparison for ALL cache sizes automatically.
Directly answers the assignment questions by plotting:
1. Throughput (Req/s)
2. Total Latency (Response Time)
3. TTFT (Prefill Time - Advanced Insight)

Usage:
    python plot.py --results-dir results --output-dir plots
"""

import os
import json
import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Matplotlib Pro Styling
# =====================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

COLORS = plt.cm.tab10.colors

# =====================================================
# File Loading (Results & Summaries)
# =====================================================
def load_result_files(results_dir):
    files = sorted(Path(results_dir).glob("*_results.json"))
    all_data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_data.append({"filename": file.name, "data": data})
    return all_data

def load_summary_files(results_dir):
    files = sorted(Path(results_dir).glob("*_summary.json"))
    summaries = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        summaries.append({"filename": file.name, "data": data})
    return summaries

def get_cache_size_int(filename):
    match = re.search(r"cache_(\d+)GB", filename)
    return int(match.group(1)) if match else -1

def get_cache_label(filename):
    size = get_cache_size_int(filename)
    return f"{size}GB Cache" if size != -1 else "0GB Cache"


# =====================================================
# OVERALL THROUGHPUT (Req / sec)
# =====================================================
def plot_throughput(summaries, output_dir):
    results = {}
    for item in summaries:
        # We use the 'single' pass as the standard baseline for throughput
        if "single" not in item["filename"]: continue
        cache = get_cache_label(item["filename"])
        results[cache] = item["data"].get("throughput_req_per_sec", 0)

    if not results: return
    
    sorted_caches = sorted(results.keys(), key=lambda c: int(c.split('G')[0]))
    throughputs = [results[c] for c in sorted_caches]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(sorted_caches, throughputs, color="#3F51B5", edgecolor="black", alpha=0.8)
    
    plt.xlabel("Local Cache Size", fontsize=11, fontweight='bold')
    plt.ylabel("Throughput (Requests / Second)", fontsize=11, fontweight='bold')
    plt.title("Overall System Throughput vs Cache Size", fontsize=14, pad=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "00_overall_throughput.png"))
    plt.close()


# =====================================================
# Q1: Latency vs Sequence Length
# =====================================================
def plot_latency_vs_length(all_data, output_dir):
    results = {}
    for item in all_data:
        if "single" not in item["filename"]: continue
        cache = get_cache_label(item["filename"])
        if cache not in results:
            results[cache] = {"x": [], "y": []}

        for row in item["data"]:
            if row.get("latency") is not None:
                results[cache]["x"].append(row["prompt_len"])
                results[cache]["y"].append(row["latency"])

    if not results: return
    sorted_caches = sorted(results.keys(), key=lambda c: int(c.split('G')[0]))

    plt.figure(figsize=(10, 6))
    for i, cache in enumerate(sorted_caches):
        plt.scatter(results[cache]["x"], results[cache]["y"], label=cache, alpha=0.7, color=COLORS[i%10])

    plt.xlabel("Sequence Length (Characters)", fontsize=11, fontweight='bold')
    plt.ylabel("Total Latency (Seconds)", fontsize=11, fontweight='bold')
    plt.title("Q1: Latency vs Sequence Length", fontsize=14, pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q1_latency_vs_length.png"))
    plt.close()


# =====================================================
# Q2: Cold vs Warm Request (Total Latency AND TTFT)
# =====================================================
def plot_cold_vs_warm(all_data, output_dir):
    results_latency = {}
    results_ttft = {}
    
    for item in all_data:
        if "repeat" not in item["filename"]: continue
        cache = get_cache_label(item["filename"])
        if cache not in results_latency:
            results_latency[cache] = {"cold": [], "warm": []}
            results_ttft[cache] = {"cold": [], "warm": []}

        for row in item["data"]:
            if row.get("pass") not in ["cold", "warm"]: continue
            
            if row.get("latency") is not None:
                results_latency[cache][row["pass"]].append(row["latency"])
            if row.get("ttft") is not None:
                results_ttft[cache][row["pass"]].append(row["ttft"])

    if not results_latency: return

    sorted_caches = sorted(results_latency.keys(), key=lambda c: int(c.split('G')[0]))
    x = np.arange(len(sorted_caches))
    width = 0.35

    # --- OPTION A: Total Latency 
    avg_cold_lat = [np.mean(results_latency[c]["cold"]) for c in sorted_caches]
    avg_warm_lat = [np.mean(results_latency[c]["warm"]) for c in sorted_caches]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, avg_cold_lat, width, label='First Request (Cold)', color="#FF5722", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, avg_warm_lat, width, label='Repeated Request (Warm)', color="#4CAF50", edgecolor="black", alpha=0.85)
    ax.set_ylabel('Total Latency / Response Time (Seconds)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Local Cache Size', fontsize=11, fontweight='bold')
    ax.set_title('Q2 Option A: Cold vs Warm (Total Latency)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_caches)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "q2_optionA_Total_Latency.png"))
    plt.close()

    avg_cold_ttft = [np.mean(results_ttft[c]["cold"]) for c in sorted_caches]
    avg_warm_ttft = [np.mean(results_ttft[c]["warm"]) for c in sorted_caches]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, avg_cold_ttft, width, label='First Request (Cold)', color="#FF9800", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, avg_warm_ttft, width, label='Repeated Request (Warm)', color="#8BC34A", edgecolor="black", alpha=0.85)
    ax.set_ylabel('Prefill Time / TTFT (Seconds)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Local Cache Size', fontsize=11, fontweight='bold')
    ax.set_title('Q2 Option B: Cold vs Warm (TTFT / Prefill Time)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_caches)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "q2_optionB_TTFT.png"))
    plt.close()


# =====================================================
# Q3: Diversity vs Performance 
# =====================================================
def plot_diversity_vs_latency(all_data, output_dir):
    results_latency = {}
    results_ttft = {}

    for item in all_data:
        if "diverse_n" not in item["filename"]: continue
        cache = get_cache_label(item["filename"])
        
        if cache not in results_latency:
            results_latency[cache] = {}
            results_ttft[cache] = {}

        try:
            n = int(item["filename"].split("diverse_n")[1][0])
        except: continue

        lats, ttfts = [], []
        for row in item["data"]:
            if row.get("latency") is not None: lats.append(row["latency"])
            if row.get("ttft") is not None: ttfts.append(row["ttft"])

        if lats: results_latency[cache][n] = np.mean(lats)
        if ttfts: results_ttft[cache][n] = np.mean(ttfts)

    if not results_latency: return
    sorted_caches = sorted(results_latency.keys(), key=lambda c: int(c.split('G')[0]))

    # --- OPTION A: Total Latency 
    plt.figure(figsize=(10, 6))
    for i, cache in enumerate(sorted_caches):
        if not results_latency[cache]: continue
        x = sorted(results_latency[cache].keys())
        y = [results_latency[cache][i] for i in x]
        plt.plot(x, y, marker="o", linewidth=2, markersize=8, label=cache, color=COLORS[i%10])

    plt.xlabel("Number of Combined Contexts (Diversity)", fontsize=11, fontweight='bold')
    plt.ylabel("Total Latency (Seconds)", fontsize=11, fontweight='bold')
    plt.title("Q3 Option A: Diversity vs Total Latency", fontsize=14, pad=15)
    plt.xticks([1, 2, 3])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q3_optionA_Total_Latency.png"))
    plt.close()

    # --- OPTION B: TTFT 
    plt.figure(figsize=(10, 6))
    for i, cache in enumerate(sorted_caches):
        if not results_ttft[cache]: continue
        x = sorted(results_ttft[cache].keys())
        y = [results_ttft[cache][i] for i in x]
        plt.plot(x, y, marker="s", linewidth=2, markersize=8, label=cache, linestyle="--", color=COLORS[i%10])

    plt.xlabel("Number of Combined Contexts (Diversity)", fontsize=11, fontweight='bold')
    plt.ylabel("Prefill Time / TTFT (Seconds)", fontsize=11, fontweight='bold')
    plt.title("Q3 Option B: Diversity vs Prefill Time (TTFT)", fontsize=14, pad=15)
    plt.xticks([1, 2, 3])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q3_optionB_TTFT.png"))
    plt.close()


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Folder containing JSON files")
    parser.add_argument("--output-dir", default="plots", help="Folder to save graphs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Reading results from: {args.results_dir}")

    all_data = load_result_files(args.results_dir)
    summaries = load_summary_files(args.results_dir)
    
    if not all_data:
        print("No result files found.")
        return

    print("Generating Throughput plot...")
    plot_throughput(summaries, args.output_dir)

    print("Generating Q1 plots...")
    plot_latency_vs_length(all_data, args.output_dir)
    
    print("Generating Q2 plots...")
    plot_cold_vs_warm(all_data, args.output_dir)
    
    print("Generating Q3 plots...")
    plot_diversity_vs_latency(all_data, args.output_dir)

    print(f"\n✓ Done. All assignment-aligned graphs saved to '{args.output_dir}/'")

if __name__ == "__main__":
    main()