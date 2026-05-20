import os
import json
import time
import torch
import torch.nn.functional as F
import multiprocessing
from pathlib import Path
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import vllm.entrypoints.openai.api_server as base_api
from vllm.entrypoints.openai.protocol import *
from fastapi import APIRouter, Request

# Sentence-transformer model used to convert text into embedding vectors.
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Absolute path to the folder containing the research paper .txt files.
DATA_DIR = Path(__file__).resolve().parent.parent / "frontend" / "data"

# RAG database
_rag_db = {}
_embed_model = None

# Marks the end of the stream
STREAM_END = "data: [DONE]"

def get_llm_embeddings(text, model):
    """
    Convert a piece of text into an embedding tensor suitable 
    for cosine-similarity comparisons.
    """

    embedding = model.encode(text, convert_to_tensor=True)
    return embedding.unsqueeze(0).cpu()

if multiprocessing.current_process().name == "MainProcess":
    print("[RAG] Loading embedding model on GPU...")
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
    print("[RAG] Pre-computing paper embeddings on GPU...")
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            key = filename.removesuffix(".txt")
            text = open(os.path.join(DATA_DIR, filename)).read()
            _rag_db[key] = (get_llm_embeddings(text, _embed_model), text)
            print(f"[RAG] Embedded: {key}")
    print("[RAG] RAG database built and ready.")

def rag_search(query):
    """
    Embed the incoming query and find the best-matching paper using
    cosine similarity against the pre-computed paper embeddings.
    """

    print(f"[RAG] Searching for: {query[:60]}...")
    q_emb = get_llm_embeddings(query, _embed_model)
    best_paper = max(_rag_db, key=lambda p: F.cosine_similarity(q_emb, _rag_db[p][0]).item())
    print(f"[RAG] Best match: {best_paper}")
    return best_paper


class BatchRequest(BaseModel):
    requests: List[ChatCompletionRequest]

class BatchedRequest(ChatCompletionRequest):
    """Used for a single request in the batch so it's possible to sort them"""
    request_id: int = -1
    context_id: str = ""


extended_router = APIRouter()

@extended_router.get("/models")
async def show_available_models(request: Request):
    """
    Proxy to vLLM's built-in /models endpoint.
    """
    
    print("v2 models is called!")
    return await base_api.show_available_models(request)

@extended_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Proxy to vLLM's standard single-request chat-completion endpoint.
    """

    print("v2 completion is called")
    return await base_api.create_chat_completion(request, raw_request)

@extended_router.post("/chat/batched-completions")
async def create_chat_batched_completions(requests: List[BatchedRequest], raw_request: Request):
    """
    Batch completion endpoint for pre-labeled requests.
    Sorts by context_id for cache locality, then processes sequentially with streaming
    enabled internally so TTFT can be measured per request.
    TTFT is recorded at the first non-empty token chunk from the stream.
    """


    # Sorting the requests based on the context_id
    requests.sort(key=lambda req: req.context_id)

    print("create_chat_completion - Batch order after sorting:")
    print([req.context_id for req in requests])

    # Collecting the results together 
    results = []
    for req in requests:
        print(f"create_chat_completion - Processing request with context_id: {req.context_id}")
        start = time.perf_counter()
        req.stream = True
        stream = await base_api.create_chat_completion(req, raw_request)

        chunks: list[str] = []
        ttft = None

        # Using stream since we need to measure the TTFT 
        async for raw_stream_chunk in stream.body_iterator:
            # Skip empty stream chunks and stop reading if end of strem
            if not raw_stream_chunk.strip():
                continue            
            if STREAM_END in raw_stream_chunk:
                break
      
            data_str = raw_stream_chunk.removeprefix("data: ").strip()
            try:
                chunk_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if not chunk_data.get("choices"):
                continue

            # Get the token and update the ttft if it's the first token
            content = chunk_data["choices"][0]["delta"].get("content", "")
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks.append(content)

        full_response = "".join(chunks)
        latency = time.perf_counter() - start

        results.append({"request_id": req.request_id, "response": full_response, "latency": latency, "ttft": ttft}) 

    return results

@extended_router.post("/chat/completions/rag")
async def create_batch_completion(batch: BatchRequest, raw_request: Request):
    """
    RAG-augmented batch inference endpoint.

    Pipeline:
      1. Classify each request: embed the question, find closest paper via cosine sim.
      2. Sort classified requests by paper name.
      3. Prepend the retrieved paper text as a user message.
      4. Run LLM inference and record per-request inference time.
      5. Return results in the ORIGINAL request order.
    """

    print(f"[SCHEDULER] Received batch of {len(batch.requests)} requests")

    # Classify each request with RAG, keep original index
    classified = []
    for idx, req in enumerate(batch.requests):
        query = req.messages[-1]["content"]
        t0 = time.time()
        paper = rag_search(query)
        retrieval_time = time.time() - t0
        classified.append((paper, retrieval_time, req, idx))

    # Reorder so same papers are grouped (cache efficiency)
    classified.sort(key=lambda x: x[0])

    # Process sequentially, writing back to original index positions
    results = [None] * len(batch.requests)
    for paper, retrieval_time, req, original_idx in classified:
        req.messages.insert(0, {"role": "user", "content": _rag_db[paper][1]})
        t0 = time.time()
        result = await base_api.create_chat_completion(req, raw_request)
        inference_time = time.time() - t0
        results[original_idx] = {
            "predicted_paper": paper,
            "retrieval_time": retrieval_time,
            "inference_time": inference_time,
        }

    return results