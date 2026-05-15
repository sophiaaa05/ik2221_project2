"""
IK2221 - Request Generator Task 3
Sends a batch of requests to the RAG scheduler and measures:
- RAG accuracy (predicted vs true context)
- Retrieval time
- Inference time
- Throughput
"""

import os
import time
import json
import random
import argparse
import httpx
import openai

IP   = "127.0.0.1"
PORT = 8000

DATA_DIR   = "lmcache-vllm-extended/frontend/data"
OUTPUT_DIR = os.environ.get("RESULTS_DIR", "results")
LABEL      = os.environ.get("RESULTS_LABEL", "task3_scheduled")

# Map context_id to a human-readable title
TITLES = {
    "vllm":                  "the vLLM paper",
    "cacheblend":            "the CacheBlend paper",
    "ServerlessLLM_summary": "the ServerlessLLM paper",
    "SplitRPC-sigmetrics23": "the SplitRPC paper",
    "aplomb-sigcomm12":      "the Aplomb paper",
    "click":                 "the Click paper",
    "metron-nsdi18":         "the Metron paper",
    "nsdi20-paper-barbette": "the Barbette paper",
    "nsdi22-paper-reda_1":   "the Reda paper",
    "osdi24-agrawal":        "the Agrawal OSDI paper",
    "osdi24-lee":            "the Lee OSDI paper",
    "osdi24-sun-biao":       "the Sun-Biao OSDI paper",
    "sigcomm2023_janus":     "the Janus paper",
    "sigcomm24-crux":        "the Crux paper",
}

# Paper-specific questions with technical keywords to improve RAG accuracy
PAPER_QUESTIONS = {
    "vllm": [
        "How does vLLM use PagedAttention to manage KV cache memory?",
        "What throughput improvements does vLLM achieve for LLM serving?",
        "How does vLLM handle memory fragmentation during inference?",
    ],
    "cacheblend": [
        "How does CacheBlend reuse KV caches across different LLM requests?",
        "What is the blending mechanism in CacheBlend for KV cache reuse?",
        "How does CacheBlend reduce prefill cost in LLM inference?",
    ],
    "ServerlessLLM_summary": [
        "How does ServerlessLLM handle cold start latency for LLM inference?",
        "What checkpoint loading optimizations does ServerlessLLM propose?",
        "How does ServerlessLLM support serverless deployment of large language models?",
    ],
    "SplitRPC-sigmetrics23": [
        "How does SplitRPC split RPC processing between CPU and GPU?",
        "What is the performance benefit of SplitRPC for network function acceleration?",
        "How does SplitRPC reduce latency in remote procedure calls?",
    ],
    "aplomb-sigcomm12": [
        "How does Aplomb consolidate middlebox functions in a cloud network?",
        "What is the architecture of Aplomb for outsourcing network middleboxes?",
        "How does Aplomb improve network management using cloud-based middleboxes?",
    ],
    "click":  [
        "How does the Click modular router enable flexible packet processing?",
        "What is the element-based architecture of the Click router?",
        "How does Click support custom network packet processing pipelines?",
    ],
    "metron-nsdi18": [
        "How does Metron achieve high-performance network function chaining?",
        "What stateful network function offloading does Metron propose?",
        "How does Metron use hardware offloading for NFV performance?",
    ],
    "nsdi20-paper-barbette": [
        "How does Barbette improve packet processing performance on multi-core systems?",
        "What abstraction does Barbette provide for high-speed packet I/O?",
        "How does Barbette achieve zero-copy packet processing?",
    ],
    "nsdi22-paper-reda_1": [
        "How does this paper address load balancing in network function virtualization?",
        "What scheduling approach is proposed for NFV service chains?",
        "How does this work improve resource efficiency in virtualized network functions?",
    ],
    "osdi24-agrawal": [
        "How does this OSDI paper improve disaggregated memory for LLM inference?",
        "What prefilling optimization is proposed in this OSDI 2024 work?",
        "How does this paper address KV cache migration in distributed LLM serving?",
    ],
    "osdi24-lee": [
        "How does this OSDI paper handle speculative decoding for LLM inference?",
        "What draft model approach is proposed in this OSDI 2024 work?",
        "How does this paper improve token generation speed in large language models?",
    ],
    "osdi24-sun-biao": [
        "How does this OSDI paper address LLM inference scheduling across GPUs?",
        "What disaggregation strategy is proposed in this OSDI 2024 work?",
        "How does this paper improve GPU utilization in LLM serving systems?",
    ],
    "sigcomm2023_janus": [
        "How does Janus handle heterogeneous network function deployment?",
        "What is the adaptive offloading mechanism proposed in Janus?",
        "How does Janus balance CPU and SmartNIC workloads for packet processing?",
    ],
    "sigcomm24-crux": [
        "How does Crux improve cross-layer network optimization?",
        "What is the resource allocation strategy proposed in the Crux paper?",
        "How does Crux handle congestion control in datacenter networks?",
    ],
}


def load_contexts(data_dir, n_docs=None):
    contexts = {}
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            context_id = filename.replace(".txt", "")
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                contexts[context_id] = f.read().strip()
            print(f"  Loaded: {context_id}")
    if n_docs is not None:
        contexts = dict(list(contexts.items())[:n_docs])
        print(f"  Using {len(contexts)} documents (--n-docs={n_docs})")
    return contexts


def build_requests(contexts):
    requests = []
    for context_id, context_text in contexts.items():
        questions = PAPER_QUESTIONS.get(context_id, [
            f"What is the main topic of {TITLES.get(context_id, context_id)}?",
            f"What problem does {TITLES.get(context_id, context_id)} address?",
            f"What methods are proposed in {TITLES.get(context_id, context_id)}?",
        ])
        for question in questions:
            requests.append({
                "context_id": context_id,
                "question":   question,
            })
    random.shuffle(requests)
    return requests


def send_batch(requests, model_id):
    """Send all requests to the RAG scheduler as a single batch.
    Only the question is sent — the server figures out the context via RAG.
    """
    batch_payload = {
        "requests": [
            {
                "model":    model_id,
                "messages": [{"role": "user", "content": req["question"]}],
                "stream":   False,
                "stop":     "\n",
            }
            for req in requests
        ]
    }

    print(f"\nSending batch of {len(requests)} requests to scheduler...")
    start = time.perf_counter()

    response = httpx.post(
        f"http://{IP}:{PORT}/v2/chat/completions/batch",
        json=batch_payload,
        timeout=2000,
    )
    response.raise_for_status()

    total_time = time.perf_counter() - start
    return total_time, response.json()


def evaluate(requests, responses):
    """Compare RAG predictions against true context_ids and measure times."""
    correct = 0
    retrieval_times = []
    inference_times = []

    print("\n=== RAG PREDICTIONS ===")
    for req, resp in zip(requests, responses):
        predicted  = resp["predicted_paper"]
        true       = req["context_id"]
        is_correct = (predicted == true)
        if is_correct:
            correct += 1
        retrieval_times.append(resp["retrieval_time"])
        inference_times.append(resp["inference_time"])
        status = "OK" if is_correct else "WRONG"
        print(f"  [{status}] true={true:30s}  predicted={predicted}")

    accuracy      = correct / len(requests) * 100
    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_inference = sum(inference_times) / len(inference_times)

    print(f"\n=== RAG EVALUATION ===")
    print(f"  Accuracy          : {correct}/{len(requests)} ({accuracy:.1f}%)")
    print(f"  Avg retrieval time: {avg_retrieval:.3f}s")
    print(f"  Avg inference time: {avg_inference:.3f}s")

    return accuracy, avg_retrieval, avg_inference


def save_results(results, summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, LABEL + "_results.json")
    summary_path = os.path.join(OUTPUT_DIR, LABEL + "_summary.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {results_path}")
    print(f"  Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="IK2221 Task 3 Request Generator")
    parser.add_argument("--n-docs", type=int, default=None,
                        help="Number of documents to use (default: all)")
    args = parser.parse_args()

    random.seed(42)

    # Connect to server
    client   = openai.OpenAI(api_key="EMPTY", base_url=f"http://{IP}:{PORT}/v2")
    model_id = client.models.list().data[0].id
    print(f"Connected to model: {model_id}")

    # Load contexts
    print(f"\nLoading contexts from {DATA_DIR}...")
    contexts = load_contexts(DATA_DIR, n_docs=args.n_docs)
    print(f"Loaded {len(contexts)} contexts.")

    # Build and send requests
    requests = build_requests(contexts)
    print(f"Built {len(requests)} requests (shuffled).")

    total_time, responses = send_batch(requests, model_id)

    # Evaluate RAG accuracy and timing
    accuracy, avg_retrieval, avg_inference = evaluate(requests, responses)

    avg_latency = total_time / len(requests)
    throughput  = len(requests) / total_time

    print(f"\n=== OVERALL PERFORMANCE ===")
    print(f"  Total time  : {total_time:.2f}s")
    print(f"  Requests    : {len(requests)}")
    print(f"  Avg latency : {avg_latency:.3f}s")
    print(f"  Throughput  : {throughput:.3f} req/s")

    # Save
    results = [
        {
            "context_id":      req["context_id"],
            "question":        req["question"],
            "predicted_paper": resp["predicted_paper"],
            "correct":         resp["predicted_paper"] == req["context_id"],
            "retrieval_time":  resp["retrieval_time"],
            "inference_time":  resp["inference_time"],
        }
        for req, resp in zip(requests, responses)
    ]

    summary = {
        "label":                  LABEL,
        "num_docs":               len(contexts),
        "num_requests":           len(requests),
        "total_time_sec":         total_time,
        "avg_latency_sec":        avg_latency,
        "throughput_req_per_sec": throughput,
        "rag_accuracy_pct":       accuracy,
        "avg_retrieval_time_sec": avg_retrieval,
        "avg_inference_time_sec": avg_inference,
    }

    save_results(results, summary)
    print("\nDone!")


if __name__ == "__main__":
    main()