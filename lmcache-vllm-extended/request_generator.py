"""
IK2221 - Request Generator 
=======================================================
Supports multiple trials, expanded contexts, and a
repeated run on the expanded set.
"""

import os
import re
from pathlib import Path
import sys
import time
import json
import random
import argparse
from datetime import datetime


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
sys.path.insert(0, FRONTEND_DIR)
from chat_session import ChatSession

BASE_DIR = Path(__file__).resolve().parent

# ─── CONFIG ───────────────────────────────────────────────────────────────────

IP   = "127.0.0.1"
PORT = 8000

SYSTEM_PROMPT = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in the document."
)

QUESTIONS = [
    "What is the main topic of this document?",
    "Write a short summary of this document.",
    "What is the key contribution described in this document?",
    "What problem does this document address?",
    "What methods or techniques are proposed in this document?",
    "What are the main conclusions or findings of this document?",
    "What limitations does this document acknowledge?",
]

# ─── LOAD CONTEXTS ────────────────────────────────────────────────────────────

def load_contexts(context_dir):
    contexts = {}
    for fname in sorted(os.listdir(context_dir)):
        if fname.endswith(".txt"):
            fpath = os.path.join(context_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            context_id = os.path.splitext(fname)[0]
            contexts[context_id] = text
            print(f"  Loaded: {context_id} ({len(text)} chars)")
    return contexts

# 
def build_request_list(contexts: dict, questions: list, seed: int) -> list:
    """One entry per (context, question) pair, shuffled with given seed."""
    pairs = [
        {
            "context_id":   ctx_id,
            "context_text": ctx_text,
            "question":     q,
            "prompt_len":   len(ctx_text) + len(q),
        }
        for ctx_id, ctx_text in contexts.items()
        for q in questions
    ]
    random.seed(seed)
    random.shuffle(pairs)
    return pairs

# ── send one request ──────────────────────────────────────────────────────────

def send_request(req):
    """
    Each request gets its own session.
    """
    session = ChatSession(IP, PORT)
    session.set_context([SYSTEM_PROMPT, req["context_text"]])

    start = time.perf_counter()
    chunks = list(session.chat(req["question"]))
    latency = time.perf_counter() - start

    response = "".join(
        c for c in chunks
        if not c.startswith("\n\n(Response delay")
    )
    return response, latency

# ── single-pass experiment ────────────────────────────────────────────────────

def run_single_pass(requests: list, pass_label: str) -> list:
    """Send every request in `requests` once, in order."""
    print(f"\n  --- {pass_label} ({len(requests)} requests) ---")
    results = []
    for i, req in enumerate(requests):
        print(f"    [{i+1:>3}/{len(requests)}] "
              f"ctx={req['context_id']:20s} "
              f"len={req['prompt_len']:>5} chars ... ",
              end="", flush=True)
        try:
            response, latency = send_request(req)
            print(f"{latency:.3f}s")
            results.append({
                "pass":       pass_label,
                "index":      i,
                "context_id": req["context_id"],
                "question":   req["question"],
                "prompt_len": req["prompt_len"],
                "latency":    latency,
                "response":   response,
                "error":      None,
            })
        except Exception as e:
            print(f"ERROR — {e}")
            results.append({
                "pass":       pass_label,
                "index":      i,
                "context_id": req["context_id"],
                "question":   req["question"],
                "prompt_len": req["prompt_len"],
                "latency":    None,
                "response":   None,
                "error":      str(e),
            })
    return results

# ── summarise results ─────────────────────────────────────────────────────────

def summarise(results: list, label: str, total_time: float) -> dict:
    successful = [r for r in results if r["latency"] is not None]
    avg_lat    = (sum(r["latency"] for r in successful) / len(successful)
                  if successful else 0.0)
    throughput = len(successful) / total_time if total_time > 0 else 0.0

    print(f"\n  ── Summary: {label} ──")
    print(f"     Requests:   {len(successful)}/{len(results)} successful")
    print(f"     Total time: {total_time:.2f}s")
    print(f"     Throughput: {throughput:.4f} req/s")
    print(f"     Avg latency:{avg_lat:.4f}s")

    return {
        "label":                  label,
        "total_time_sec":         total_time,
        "num_requests":           len(results),
        "num_successful":         len(successful),
        "throughput_req_per_sec": throughput,
        "avg_latency_sec":        avg_lat,
    }


def get_trial_seed(base_seed: int, trial: int) -> int:
    """Return a unique seed for a trial, keeping pseudo‑independence."""
    return base_seed + trial * 1000   # arbitrary offset


def experiment_single(contexts, questions, cache_label, output_dir, trial=1):
    label = f"{cache_label}_single_trial{trial}"
    seed = get_trial_seed(42, trial)
    requests = build_request_list(contexts, questions, seed)

    print(f"\n{'='*65}")
    print(f"RUN A — Single Pass   |  cache={cache_label}  |  trial {trial}")
    print(f"{'='*65}")

    t0      = time.perf_counter()
    results = run_single_pass(requests, pass_label="pass1_cold")
    total   = time.perf_counter() - t0

    summary = summarise(results, label, total)
    save(results, summary, label, output_dir)
    return results, summary


def experiment_repeat(contexts, questions, cache_label, output_dir, trial=1, run_type="repeat"):
    label = f"{cache_label}_{run_type}_trial{trial}"
    base_seed = 42 if run_type == "repeat" else 123   # different base for diverse repeat
    seed = get_trial_seed(base_seed, trial)
    requests = build_request_list(contexts, questions, seed)

    print(f"\n{'='*65}")
    print(f"RUN B — Repeat (Cache-Hit)  |  cache={cache_label}  |  run={run_type}  |  trial {trial}")
    print(f"{'='*65}")
    print("  Pass 1: cold cache ")
    print("  Pass 2: warm cache ")

    t0 = time.perf_counter()
    results_p1 = run_single_pass(requests, pass_label="pass1_cold")
    results_p2 = run_single_pass(requests, pass_label="pass2_warm")
    total = time.perf_counter() - t0

    all_results = results_p1 + results_p2
    summary     = summarise(all_results, label, total)

    ok1 = [r for r in results_p1 if r["latency"] is not None]
    ok2 = [r for r in results_p2 if r["latency"] is not None]
    if ok1 and ok2:
        avg1 = sum(r["latency"] for r in ok1) / len(ok1)
        avg2 = sum(r["latency"] for r in ok2) / len(ok2)
        improvement = (avg1 - avg2) / avg1 * 100
        print(f"\n  Cache-hit result:")
        print(f"     Avg latency pass 1 (cold): {avg1:.4f}s")
        print(f"     Avg latency pass 2 (warm): {avg2:.4f}s")
        print(f"     Improvement:               {improvement:.1f}%")

    save(all_results, summary, label, output_dir)
    return all_results, summary


def experiment_diverse(contexts, questions, cache_label, output_dir, trial=1, run_type="diverse"):
    label = f"{cache_label}_{run_type}_trial{trial}"
    seed = get_trial_seed(123, trial)
    requests = build_request_list(contexts, questions, seed)

    print(f"\n{'='*65}")
    print(f"RUN C — {run_type.upper()}  |  cache={cache_label}  |  trial {trial}")
    print(f"{'='*65}")

    t0      = time.perf_counter()
    results = run_single_pass(requests, pass_label=f"pass1_{run_type}")
    total   = time.perf_counter() - t0

    summary = summarise(results, label, total)
    save(results, summary, label, output_dir)
    return results, summary

# ── save helpers ──────────────────────────────────────────────────────────────

def save(results: list, summary: dict, label: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_")

    rpath = os.path.join(output_dir, f"{safe_label}_{ts}_results.json")
    spath = os.path.join(output_dir, f"{safe_label}_{ts}_summary.json")

    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved results → {rpath}")
    print(f"  Saved summary → {spath}")


def save_averaged_summary(avg_summary: dict, label: str, output_dir: str):
    """Save a final averaged summary (no trial number)."""
    os.makedirs(output_dir, exist_ok=True)
    spath = os.path.join(output_dir, f"{label}_avg_summary.json")
    with open(spath, "w") as f:
        json.dump(avg_summary, f, indent=2)
    print(f"  Saved averaged summary → {spath}")

# ── average trial summaries ───────────────────────────────────────────────────

def average_trial_summaries(summary_list: list, base_label: str) -> dict:
    first = summary_list[0]
    n = len(summary_list)
    avg = {
        "label": base_label,
        "num_requests": first["num_requests"],
        "num_successful": first["num_successful"],
        "total_time_sec": sum(s["total_time_sec"] for s in summary_list) / n,
        "throughput_req_per_sec": sum(s["throughput_req_per_sec"] for s in summary_list) / n,
        "avg_latency_sec": sum(s["avg_latency_sec"] for s in summary_list) / n,
    }
    return avg

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IK2221 Request Generator")
    parser.add_argument("--context-dir", type=Path, default=BASE_DIR / "frontend" / "data",
                        help="Normal context folder")
    parser.add_argument("--context-dir-expanded", type=Path,
                        default=BASE_DIR / "frontend" / "data_expanded",
                        help="Expanded context folder for high‑diversity runs")
    parser.add_argument("--mode", choices=["single", "repeat", "diverse",
                                           "diverse_more_contexts",
                                           "diverse_more_repeat",
                                           "all"],
                        default="all")
    parser.add_argument("--cache-label", default="cache_XGB")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per experiment")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  IK2221 Request Generator")
    print(f"  Cache label: {args.cache_label}")
    print(f"  Normal ctx : {args.context_dir}")
    print(f"  Expanded ctx: {args.context_dir_expanded}")
    print(f"  Mode       : {args.mode}")
    print(f"  Trials     : {args.trials}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'='*65}")

    # Load normal contexts
    contexts = load_contexts(args.context_dir)
    print(f"\nLoaded {len(contexts)} normal context file(s).")

    # Load expanded contexts if needed
    contexts_expanded = None
    if args.mode in ("diverse_more_contexts", "diverse_more_repeat", "all"):
        if args.context_dir_expanded.exists():
            contexts_expanded = load_contexts(args.context_dir_expanded)
            print(f"Loaded {len(contexts_expanded)} expanded context file(s).")
        else:
            print("WARNING: expanded context dir not found; skipping those runs.")
            # Disable those modes
            if args.mode == "diverse_more_contexts":
                return
            if args.mode == "diverse_more_repeat":
                return

    # Helper to run an experiment multiple trials and save averaged summary
    def run_trials(exp_func, ctx, qs, label, run_type=None):
        trial_summaries = []
        for trial in range(1, args.trials+1):
            if run_type is None:
                res, summ = exp_func(ctx, qs, args.cache_label, args.output_dir, trial=trial)
            else:
                res, summ = exp_func(ctx, qs, args.cache_label, args.output_dir,
                                     trial=trial, run_type=run_type)
            trial_summaries.append(summ)
        avg = average_trial_summaries(trial_summaries, f"{args.cache_label}_{run_type if run_type else 'single'}_avg")
        save_averaged_summary(avg, f"{args.cache_label}_{run_type if run_type else 'single'}_avg", args.output_dir)

    # Execute requested modes
    if args.mode in ("single", "all"):
        run_trials(experiment_single, contexts, QUESTIONS, "single")

    if args.mode in ("repeat", "all"):
        run_trials(experiment_repeat, contexts, QUESTIONS, "repeat", run_type="repeat")

    if args.mode in ("diverse", "all"):
        run_trials(experiment_diverse, contexts, QUESTIONS, "diverse", run_type="diverse")

    if args.mode in ("diverse_more_contexts", "all") and contexts_expanded:
        run_trials(experiment_diverse, contexts_expanded, QUESTIONS, "diverse_more_contexts",
                   run_type="diverse_more_contexts")

    if args.mode in ("diverse_more_repeat", "all") and contexts_expanded:
        run_trials(experiment_repeat, contexts_expanded, QUESTIONS, "diverse_more_repeat",
                   run_type="diverse_more_repeat")

    print(f"\n✓ Done. Results saved in '{args.output_dir}/'")

if __name__ == "__main__":
    main()