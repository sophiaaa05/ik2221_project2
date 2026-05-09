# Request generator for task 1

"""
IK2221 - Task 1: Request Generator
====================================
Loads context files (.txt), generates questions for each,
shuffles them (unordered), sends them sequentially to the LLM,
and records latency + throughput for analysis.


"""


import os
import time
import json
import random
import argparse
import requests
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────

LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


QUESTIONS = [
    "What is the main topic of this document?",
    "Write a short summary of this document.",
    "What is the key contribution described in this document?",
    "What problem does this document address?",
    "What methods or techniques are proposed in this document?",

    
    "What datasets or benchmarks are used in this document?",
    "What evaluation metrics are used?",
    "What is the system architecture described in this document?",
    "What hardware or infrastructure is required?",
    "What are the computational costs mentioned?",
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
    """
    Create a list of request dicts, one per (context, question) pair.
    repeat=2 means each pair appears twice 
    """
    requests_list = []
    for context_id, context_text in contexts.items():
        for question in questions:
            for _ in range(repeat):
                prompt = f"{context_text}\n\nQuestion: {question}"
                requests_list.append({
                    "context_id": context_id,
                    "question": question,
                    "prompt": prompt,
                    "prompt_len": len(prompt),
                })
    random.shuffle(requests_list)
    return requests_list

# ─── SEND REQUEST ─────────────────────────────────────────────────────────────

def send_request(prompt, timeout=120):
    """Send prompt to LLM, return (response_text, latency_in_seconds)."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    resp = requests.post(LLM_URL, json=payload, timeout=timeout)
    latency = time.perf_counter() - start
    resp.raise_for_status()
    response_text = resp.json()["choices"][0]["message"]["content"]
    return response_text, latency

# ─── RUN EXPERIMENT ───────────────────────────────────────────────────────────

def run_experiment(requests_list, label):
    """Send all requests sequentially. Returns (results list, summary dict)."""
    print(f"\n{'='*60}")
    print(f"Experiment: {label}  |  {len(requests_list)} requests")
    print(f"{'='*60}")

    results = []
    experiment_start = time.perf_counter()

    for i, req in enumerate(requests_list):
        print(f"  [{i+1}/{len(requests_list)}] context={req['context_id']} "
              f"prompt_len={req['prompt_len']} chars ... ", end="", flush=True)
        try:
            response_text, latency = send_request(req["prompt"])
            print(f"{latency:.2f}s")
            results.append({
                "index": i,
                "context_id": req["context_id"],
                "question": req["question"],
                "prompt_len": req["prompt_len"],
                "latency": latency,
                "response": response_text,
                "error": None,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "index": i,
                "context_id": req["context_id"],
                "question": req["question"],
                "prompt_len": req["prompt_len"],
                "latency": None,
                "response": None,
                "error": str(e),
            })

    total_time = time.perf_counter() - experiment_start
    successful = [r for r in results if r["latency"] is not None]
    throughput = len(successful) / total_time if total_time > 0 else 0
    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0

    print(f"\n--- Summary: {label} ---")
    print(f"  Total time:   {total_time:.2f}s")
    print(f"  Successful:   {len(successful)}/{len(results)}")
    print(f"  Throughput:   {throughput:.3f} req/s")
    print(f"  Avg latency:  {avg_latency:.2f}s")

    summary = {
        "label": label,
        "total_time_sec": total_time,
        "num_requests": len(results),
        "num_successful": len(successful),
        "throughput_req_per_sec": throughput,
        "avg_latency_sec": avg_latency,
    }
    return results, summary

# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────

def save_results(results, summary, label, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_")

    results_path = os.path.join(output_dir, f"{safe_label}_{timestamp}_results.json")
    summary_path = os.path.join(output_dir, f"{safe_label}_{timestamp}_summary.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {results_path}")
    print(f"  Saved: {summary_path}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IK2221 Task 1 - Request Generator")
    parser.add_argument("--context-dir", default="../ik2221-data-resources",
                        help="Path to folder with .txt context files")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each (context, question) pair N times. "
                             "Use 2 to test cache hit on repeated requests.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible shuffling")
    parser.add_argument("--output-dir", default="results",
                        help="Folder to save JSON result files")
    parser.add_argument("--label", default="experiment",
                        help="Label for this run")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\nLoading contexts from: {args.context_dir}")
    contexts = load_contexts(args.context_dir)
    print(f"Loaded {len(contexts)} context files.")

    requests_list = generate_requests(contexts, QUESTIONS, repeat=args.repeat)
    print(f"Generated {len(requests_list)} requests (shuffled).")

    results, summary = run_experiment(requests_list, label=args.label)

    print(f"\nSaving results to: {args.output_dir}/")
    save_results(results, summary, label=args.label, output_dir=args.output_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()