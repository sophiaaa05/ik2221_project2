"""
IK2221 - Request Generator
=======================================================
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

from openai import OpenAI

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))

# ── config ────────────────────────────────────────────────────────────────────
IP   = "127.0.0.1"
PORT = 8000

SYSTEM_PROMPT = (
    "You are a helpful assistant. I will now give you a document and "
    "please answer my question afterwards based on the content in the document."
)

CONTEXT_SEPARATOR = "###"

QUESTIONS = [
    "What is the main topic of this document?",
    "Write a short summary of this document.",
    "What is the key contribution described in this document?",
    "What problem does this document address?",
    "What methods or techniques are proposed in this document?",
    "What are the main conclusions or findings of this document?",
    "What limitations does this document acknowledge?",
]

# ── shared OpenAI client ──────────────────────────────────────────────────────
# Created once so model listing overhead is paid only at startup,
# not on every request (which would inflate latency measurements).

def make_client(ip: str, port: int) -> tuple:
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{ip}:{port}/v2",
    )
    model_id = client.models.list().data[0].id
    print(f"  Connected to model: {model_id}")
    return client, model_id

# ── load contexts ─────────────────────────────────────────────────────────────
def load_contexts(context_dir: Path) -> dict:
    contexts = {}
    for fname in sorted(os.listdir(context_dir)):
        if fname.endswith(".txt"):
            fpath = context_dir / fname
            text  = fpath.read_text(encoding="utf-8").strip()
            ctx_id = fpath.stem
            contexts[ctx_id] = text
            print(f"  Loaded: {ctx_id} ({len(text)} chars)")
    return contexts

# ── request list builders ─────────────────────────────────────────────────────
def build_request_list(contexts: dict, questions: list, seed: int) -> list:
    """One entry per (context, question) pair, shuffled."""
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


def build_diverse_request_list(contexts: dict, questions: list, seed: int,
                                n_contexts: int) -> list:
    ctx_items = list(contexts.items())
    if len(ctx_items) < n_contexts:
        return []

    random.seed(seed)
    pairs = []
    
    # ADD THIS LOOP so it generates 98 requests instead of 7
    num_iterations = len(ctx_items) 
    for _ in range(num_iterations):
        for q in questions:
            chosen   = random.sample(ctx_items, n_contexts)
            ctx_id   = "+".join(c[0] for c in chosen)
            ctx_text = ("\n\n" + CONTEXT_SEPARATOR + "\n\n").join(c[1] for c in chosen)
            pairs.append({
                "context_id":   ctx_id,
                "context_text": ctx_text,
                "question":     q,
                "prompt_len":   len(ctx_text) + len(q),
            })
            
    random.shuffle(pairs)
    return pairs

# ── send one request ──────────────────────────────────────────────────────────
def send_request(req: dict, client: OpenAI, model_id: str) -> tuple:
    """
    Build the full message history and stream the response.
    Latency = wall-clock time from first API call to last token received.
    Time to first token (TTFT) = time from first API call to receiving the first token.

    """
    context_prime = CONTEXT_SEPARATOR.join([SYSTEM_PROMPT, req["context_text"]])
    messages = [
        {"role": "user",      "content": context_prime},
        {"role": "assistant", "content": "Got it!"},
        {"role": "user",      "content": req["question"]},
    ]

    start = time.perf_counter()
    ttft  = None  # time to first token

    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.5,
        stream=True,
        stop="\n",
    )

    chunks = []
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            if ttft is None:
                ttft = time.perf_counter() - start
            chunks.append(content)

    latency  = time.perf_counter() - start
    response = "".join(chunks)
    return response, latency, ttft

# ── single pass ───────────────────────────────────────────────────────────────
def run_single_pass(requests: list, pass_label: str,
                    client: OpenAI, model_id: str) -> list:
    """Send every request once, in order."""
    print(f"\n  --- {pass_label} ({len(requests)} requests) ---")
    results = []
    for i, req in enumerate(requests):
        print(f"    [{i+1:>3}/{len(requests)}] "
              f"ctx={req['context_id'][:20]:20s} "
              f"len={req['prompt_len']:>6} chars ... ",
              end="", flush=True)
        try:
            response, latency, ttft = send_request(req, client, model_id)
            print(f"{latency:.3f}s  (ttft={ttft:.3f}s)")
            results.append({
                "pass":       pass_label,
                "index":      i,
                "context_id": req["context_id"],
                "question":   req["question"],
                "prompt_len": req["prompt_len"],
                "latency":    latency,
                "ttft":       ttft,
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
                "ttft":       None,
                "response":   None,
                "error":      str(e),
            })
    return results

# ── repeat pass (cold → warm per request) ────────────────────────────────────
def run_repeat_pass(requests: list, client: OpenAI, model_id: str) -> list:
    """
    For each request, send it twice back-to-back.
    First send  → cold  (no KV-cache hit expected)
    Second send → warm  (KV-cache should reuse the context prefix)

    """
    print(f"\n  --- repeat cold->warm ({len(requests)} requests) ---")
    results = []
    for i, req in enumerate(requests):
        print(f"    [{i+1:>3}/{len(requests)}] "
              f"ctx={req['context_id'][:20]:20s} "
              f"len={req['prompt_len']:>6} chars",
              end="", flush=True)

        for pass_label in ("cold", "warm"):
            try:
                response, latency, ttft = send_request(req, client, model_id)
                print(f"  {pass_label}={latency:.3f}s", end="", flush=True)
                results.append({
                    "pass":       pass_label,
                    "index":      i,
                    "context_id": req["context_id"],
                    "question":   req["question"],
                    "prompt_len": req["prompt_len"],
                    "latency":    latency,
                    "ttft":       ttft,
                    "response":   response,
                    "error":      None,
                })
            except Exception as e:
                print(f"  {pass_label}=ERROR({e})", end="", flush=True)
                results.append({
                    "pass":       pass_label,
                    "index":      i,
                    "context_id": req["context_id"],
                    "question":   req["question"],
                    "prompt_len": req["prompt_len"],
                    "latency":    None,
                    "ttft":       None,
                    "response":   None,
                    "error":      str(e),
                })
        print()
    return results

# ── summarise ─────────────────────────────────────────────────────────────────
def summarise(results: list, label: str, total_time: float) -> dict:
    ok  = [r for r in results if r["latency"] is not None]
    avg = sum(r["latency"] for r in ok) / len(ok) if ok else 0.0
    thr = len(ok) / total_time if total_time > 0 else 0.0

    print(f"\n  -- Summary: {label} --")
    print(f"     Requests   : {len(ok)}/{len(results)} successful")
    print(f"     Total time : {total_time:.2f}s")
    print(f"     Throughput : {thr:.4f} req/s")
    print(f"     Avg latency: {avg:.4f}s")

    summary = {
        "label":                  label,
        "total_time_sec":         total_time,
        "num_requests":           len(results),
        "num_successful":         len(ok),
        "throughput_req_per_sec": thr,
        "avg_latency_sec":        avg,
    }

    cold = [r for r in ok if r["pass"] == "cold"]
    warm = [r for r in ok if r["pass"] == "warm"]
    if cold and warm:
        avg_cold   = sum(r["latency"] for r in cold) / len(cold)
        avg_warm   = sum(r["latency"] for r in warm) / len(warm)
        improvement = (avg_cold - avg_warm) / avg_cold * 100
        print(f"     Avg latency (cold): {avg_cold:.4f}s")
        print(f"     Avg latency (warm): {avg_warm:.4f}s")
        print(f"     Cache improvement : {improvement:.1f}%")
        summary.update({
            "avg_latency_cold_sec":  avg_cold,
            "avg_latency_warm_sec":  avg_warm,
            "cache_improvement_pct": improvement,
        })

    return summary

# ── experiments ───────────────────────────────────────────────────────────────
def experiment_single(contexts, questions, cache_label, output_dir,
                      client, model_id):
    label    = f"{cache_label}_single"
    requests = build_request_list(contexts, questions, seed=42)

    print(f"\n{'='*65}")
    print(f"RUN A — Single Pass  |  cache={cache_label}")
    print(f"  {len(requests)} requests  ({len(contexts)} contexts × {len(questions)} questions)")
    print(f"{'='*65}")

    t0      = time.perf_counter()
    results = run_single_pass(requests, pass_label="cold", client=client, model_id=model_id)
    total   = time.perf_counter() - t0

    summary = summarise(results, label, total)
    save(results, summary, label, output_dir)
    return results, summary


def experiment_repeat(contexts, questions, cache_label, output_dir,
                      client, model_id):
    label    = f"{cache_label}_repeat"
    requests = build_request_list(contexts, questions, seed=42)

    print(f"\n{'='*65}")
    print(f"RUN B — Repeat cold→warm  |  cache={cache_label}")
    print(f"  {len(requests)} requests × 2 sends each")
    print(f"{'='*65}")

    t0      = time.perf_counter()
    results = run_repeat_pass(requests, client=client, model_id=model_id)
    total   = time.perf_counter() - t0

    summary = summarise(results, label, total)
    save(results, summary, label, output_dir)
    return results, summary


def experiment_diverse(contexts, questions, cache_label, output_dir,
                       client, model_id, n_contexts: int):
    label    = f"{cache_label}_diverse_n{n_contexts}"
    requests = build_diverse_request_list(contexts, questions, seed=42,
                                          n_contexts=n_contexts)
    if not requests:
        return [], {}

    print(f"\n{'='*65}")
    print(f"RUN C — Diverse (n={n_contexts} contexts/request)  |  cache={cache_label}")
    print(f"  {len(requests)} requests")
    print(f"{'='*65}")

    t0      = time.perf_counter()
    results = run_single_pass(requests, pass_label="cold", client=client, model_id=model_id)
    total   = time.perf_counter() - t0

    summary = summarise(results, label, total)
    save(results, summary, label, output_dir)
    return results, summary

# ── save ──────────────────────────────────────────────────────────────────────
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

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IK2221 Request Generator")
    parser.add_argument("--context-dir", type=Path,
                        default=BASE_DIR / "frontend" / "data",
                        help="Folder of .txt context files")
    parser.add_argument("--mode",
                        choices=["single", "repeat", "diverse_sweep", "all"],
                        default="all",
                        help=(
                            "single         – Q1: latency vs length\n"
                            "repeat         – Q2: cold vs warm (KV cache)\n"
                            "diverse_sweep  – Q3: n=1..max-contexts contexts\n"
                            "all            – run all three"
                        ))
    parser.add_argument("--cache-label", default="cache_default",
                        help="Label for this cache config (e.g. cache_2GB). "
                             "Used in filenames so runs with different cache "
                             "sizes are kept separate.")
    parser.add_argument("--max-contexts", type=int, default=3,
                        help="Max number of contexts per request in diverse_sweep "
                             "(sweeps from n=1 up to this value, default: 3)")
    parser.add_argument("--output-dir", default="results",
                        help="Folder to write result JSON files (default: results/)")
    parser.add_argument("--ip",   default=IP)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  IK2221 Request Generator")
    print(f"  Cache label   : {args.cache_label}")
    print(f"  Context dir   : {args.context_dir}")
    print(f"  Mode          : {args.mode}")
    print(f"  Max contexts  : {args.max_contexts}  (diverse_sweep only)")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Server        : {args.ip}:{args.port}")
    print(f"{'='*65}")

    print("\nConnecting to vLLM server …")
    client, model_id = make_client(args.ip, args.port)

    print(f"\nLoading contexts from '{args.context_dir}' …")
    contexts = load_contexts(args.context_dir)
    print(f"Loaded {len(contexts)} context file(s).")

    if args.mode in ("single", "all"):
        experiment_single(contexts, QUESTIONS, args.cache_label, args.output_dir,
                          client, model_id)

    if args.mode in ("repeat", "all"):
        experiment_repeat(contexts, QUESTIONS, args.cache_label, args.output_dir,
                          client, model_id)

    if args.mode in ("diverse_sweep", "all"):
        max_n = min(args.max_contexts, len(contexts))
        for n in range(1, max_n + 1):
            experiment_diverse(contexts, QUESTIONS, args.cache_label, args.output_dir,
                               client, model_id, n_contexts=n)

    print(f"\n✓ Done. Results saved in '{args.output_dir}/'")


if __name__ == "__main__":
    main()