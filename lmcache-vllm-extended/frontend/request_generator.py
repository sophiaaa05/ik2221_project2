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
from datetime import datetime


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "lmcache-vllm-extended", "frontend")
sys.path.insert(0, FRONTEND_DIR)
from chat_session import ChatSession

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

# ─── GENERATE REQUESTS ────────────────────────────────────────────────────────

def generate_requests(contexts, questions, repeat=1):
    """Generate shuffled list of (context_id, context_text, question) tuples."""
    requests_list = []
    for context_id, context_text in contexts.items():
        for question in questions:
            for _ in range(repeat):
                requests_list.append({
                    "context_id":   context_id,
                    "context_text": context_text,
                    "question":     question,
                    "prompt_len":   len(context_text) + len(question),
                })
    random.shuffle(requests_list)
    return requests_list


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


def run_experiment(requests_list, label):
    """Send all requests sequentially."""
    print(f"\n{'='*60}")
    print(f"Experiment: {label}  |  {len(requests_list)} requests")
    print(f"{'='*60}")

    results = []
    experiment_start = time.perf_counter()

    for i, req in enumerate(requests_list):
        print(f"  [{i+1}/{len(requests_list)}] "
              f"context={req['context_id']} "
              f"prompt_len={req['prompt_len']} ... ",
              end="", flush=True)
        try:
            response, latency = send_request(req)
            print(f"{latency:.2f}s")
            results.append({
                "index":      i,
                "context_id": req["context_id"],
                "question":   req["question"],
                "prompt_len": req["prompt_len"],
                "latency":    latency,
                "response":   response,
                "error":      None,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "index":      i,
                "context_id": req["context_id"],
                "question":   req["question"],
                "prompt_len": req["prompt_len"],
                "latency":    None,
                "response":   None,
                "error":      str(e),
            })

    total_time = time.perf_counter() - experiment_start
    successful = [r for r in results if r["latency"] is not None]
    throughput  = len(successful) / total_time if total_time > 0 else 0
    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0

    print(f"\n--- Summary: {label} ---")
    print(f"  Total time:   {total_time:.2f}s")
    print(f"  Successful:   {len(successful)}/{len(results)}")
    print(f"  Throughput:   {throughput:.3f} req/s")
    print(f"  Avg latency:  {avg_latency:.2f}s")

    summary = {
        "label":                  label,
        "total_time_sec":         total_time,
        "num_requests":           len(results),
        "num_successful":         len(successful),
        "throughput_req_per_sec": throughput,
        "avg_latency_sec":        avg_latency,
    }
    return results, summary


def run_batched(requests_list, batch_size, label):
    """
    Split requests into batches and send each batch to the scheduler.
    """
    print(f"\n{'='*60}")
    print(f"Batched experiment: {label}")
    print(f"Total requests: {len(requests_list)}  |  Batch size: {batch_size}")
    print(f"{'='*60}")

    # Split into batches
    batches = [
        requests_list[i:i+batch_size]
        for i in range(0, len(requests_list), batch_size)
    ]
    print(f"  Number of batches: {len(batches)}")

    all_results = []
    experiment_start = time.perf_counter()

    for b_idx, batch in enumerate(batches):
        print(f"\n  Batch {b_idx+1}/{len(batches)} ({len(batch)} requests):")
        for i, req in enumerate(batch):
            print(f"    [{i+1}/{len(batch)}] "
                  f"context={req['context_id']} ... ",
                  end="", flush=True)
            try:
                response, latency = send_request(req)
                print(f"{latency:.2f}s")
                all_results.append({
                    "index":      len(all_results),
                    "batch":      b_idx,
                    "context_id": req["context_id"],
                    "question":   req["question"],
                    "prompt_len": req["prompt_len"],
                    "latency":    latency,
                    "response":   response,
                    "error":      None,})
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "index":      len(all_results),
                    "batch":      b_idx,
                    "context_id": req["context_id"],
                    "question":   req["question"],
                    "prompt_len": req["prompt_len"],
                    "latency":    None,
                    "response":   None,
                    "error":      str(e), })

    total_time  = time.perf_counter() - experiment_start
    successful  = [r for r in all_results if r["latency"] is not None]
    throughput  = len(successful) / total_time if total_time > 0 else 0
    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0

    print(f"\n--- Summary: {label} ---")
    print(f"  Total time:   {total_time:.2f}s")
    print(f"  Successful:   {len(successful)}/{len(all_results)}")
    print(f"  Throughput:   {throughput:.3f} req/s")
    print(f"  Avg latency:  {avg_latency:.2f}s")

    summary = {
        "label":                  label,
        "batch_size":             batch_size,
        "num_batches":            len(batches),
        "total_time_sec":         total_time,
        "num_requests":           len(all_results),
        "num_successful":         len(successful),
        "throughput_req_per_sec": throughput,
        "avg_latency_sec":        avg_latency,
    }
    return all_results, summary


def save_results(results, summary, label, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_")

    results_path = os.path.join(output_dir, f"{safe_label}_{timestamp}_results.json")
    summary_path = os.path.join(output_dir, f"{safe_label}_{timestamp}_summary.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {results_path}")
    print(f"  Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="IK2221 Request Generator ")
    parser.add_argument("--context-dir", default="../../ik2221-data-resources",
                        help="Path to folder with .txt context files")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each (context, question) pair N times")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--output-dir", default="results",
                        help="Folder to save result JSON files")
    parser.add_argument("--label", default="experiment",
                        help="Label for this run")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size for scheduler.")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\nLoading contexts from: {args.context_dir}")
    contexts = load_contexts(args.context_dir)
    print(f"Loaded {len(contexts)} context files.")

    requests_list = generate_requests(contexts, QUESTIONS, repeat=args.repeat)
    print(f"Generated {len(requests_list)} requests (shuffled).")

    if args.batch_size > 0:
        
        results, summary = run_batched(requests_list, args.batch_size, label=args.label)
    else:
        
        results, summary = run_experiment(requests_list, label=args.label)

    print(f"\nSaving results to: {args.output_dir}/")
    save_results(results, summary, label=args.label, output_dir=args.output_dir)
    print("\nDone!")

if __name__ == "__main__":
    main()