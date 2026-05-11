"""
IK2221 - Request Generator 
=======================================================

"""

import os
from pathlib import Path
import sys
import time
import json
import random
import argparse
from datetime import datetime


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "lmcache-vllm-extended", "frontend")
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

QUESTIONS_DIVERSE = [
    "What is the main topic of this document?",
    "Write a short summary of this document.",
    "What is the key contribution described in this document?",
    "What problem does this document address?",
    "What methods or techniques are proposed in this document?",
    "What are the main conclusions or findings of this document?",
    "What limitations does this document acknowledge?",
    "What assumptions does this document make?",
    "What datasets or benchmarks are used in this document?",
    "How does this document evaluate its proposed solution?",
    "What baseline methods does this document compare against?",
    "What are the computational requirements described in this document?",
    "What future work does this document suggest?",
    "In one sentence, what is this document about?",
    "Name the three most important concepts in this document.",
    "What is the biggest open question left by this document?",
]

# ── load contexts ─────────────────────────────────────────────────────────────

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

# ── request list builders ─────────────────────────────────────────────────────

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
    """
    Send every request in `requests` once, in order.
    Returns list of result dicts.
    """
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


def experiment_single(contexts, questions, cache_label, output_dir):
    """
    """
    label    = f"{cache_label}_single"
    requests = build_request_list(contexts, questions, seed=42)

    print(f"\n{'='*65}")
    print(f"RUN A — Single Pass   |  cache={cache_label}")
    print(f"{'='*65}")

    t0      = time.perf_counter()
    results = run_single_pass(requests, pass_label="pass1_cold")
    total   = time.perf_counter() - t0

    summary = summarise(results, label, total)
    save(results, summary, label, output_dir)
    return results, summary


def experiment_repeat(contexts, questions, cache_label, output_dir):

    label    = f"{cache_label}_repeat"
    requests = build_request_list(contexts, questions, seed=42)

    print(f"\n{'='*65}")
    print(f"RUN B — Repeat (Cache-Hit)  |  cache={cache_label}")
    print(f"{'='*65}")
    print("  Pass 1: cold cache ")
    print("  Pass 2: warm cache ")

    t0 = time.perf_counter()

    results_p1 = run_single_pass(requests, pass_label="pass1_cold")

    results_p2 = run_single_pass(requests, pass_label="pass2_warm")

    total = time.perf_counter() - t0

    all_results = results_p1 + results_p2
    summary     = summarise(all_results, label, total)

    # extra per-pass summaries printed for convenience
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


def experiment_diverse(contexts, questions, cache_label, output_dir):

    label    = f"{cache_label}_diverse"
    requests = build_request_list(contexts, questions, seed=123)

    print(f"\n{'='*65}")
    print(f"RUN C — Diverse (seed=123)  |  cache={cache_label}")
    print(f"{'='*65}")

    t0      = time.perf_counter()
    results = run_single_pass(requests, pass_label="pass1_diverse")
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

def main():

    parser = argparse.ArgumentParser(description="IK2221 Request Generator")
    parser.add_argument("--context-dir", type=Path, default=BASE_DIR / "frontend" / "data")
    parser.add_argument("--mode", choices=["single", "repeat", "diverse", "all"], default="all")
    parser.add_argument("--cache-label", default="cache_XGB")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  IK2221 Request Generator")
    print(f"  Mode:        {args.mode}")
    print(f"  Cache label: {args.cache_label}")
    print(f"  Context dir: {args.context_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"{'='*65}")

    contexts = load_contexts(args.context_dir)
    print(f"\nLoaded {len(contexts)} context file(s).")

    if args.mode in ("single", "all"):
        experiment_single(contexts, QUESTIONS, args.cache_label, args.output_dir)

    if args.mode in ("repeat", "all"):
        experiment_repeat(contexts, QUESTIONS, args.cache_label, args.output_dir)

    if args.mode in ("diverse", "all"):
        experiment_diverse(contexts, QUESTIONS_DIVERSE, args.cache_label, args.output_dir)

    print(f"\n✓ Done. Results saved in '{args.output_dir}/'")

if __name__ == "__main__":
    main()