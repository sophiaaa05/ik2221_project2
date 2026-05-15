import os
import time
import torch
import torch.nn.functional as F
from typing import List
from sentence_transformers import SentenceTransformer
import vllm.entrypoints.openai.api_server as base_api
from vllm.entrypoints.openai.protocol import *
from fastapi import APIRouter, Request
<<<<<<< Updated upstream
import json
=======
from pydantic import BaseModel
import multiprocessing
>>>>>>> Stashed changes

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # best general-purpose sentence transformer
DATA_DIR = "/home/jovyan/ik2221_project2/lmcache-vllm-extended/frontend/data/"

_rag_db = {}
_embed_model = None

def get_llm_embeddings(text, model):
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
    print(f"[RAG] Searching for: {query[:60]}...")
    q_emb = get_llm_embeddings(query, _embed_model)
    best_paper = max(_rag_db, key=lambda p: F.cosine_similarity(q_emb, _rag_db[p][0]).item())
    print(f"[RAG] Best match: {best_paper}")
    return best_paper

class BatchRequest(BaseModel):
    requests: List[ChatCompletionRequest]

extended_router = APIRouter()

class BatchedRequest(ChatCompletionRequest):
    """Used for a single request in the batch so it's possible to sort them"""
    request_id: int = -1
    context_id: str = ""

@extended_router.get("/models")
async def show_available_models(request: Request):
    print("v2 models is called!")
    return await base_api.show_available_models(request)

@extended_router.post("/chat/completions")
async def create_chat_completion(requests: List[BatchedRequest], raw_request: Request):
    print("v2 completion is called")
<<<<<<< Updated upstream
    # Sorting the requests based on the context_id
    requests.sort(key=lambda req: req.context_id)

    print("Batch order after sorting:")
    print([req.context_id for req in requests])

    # Collecting the results together 
    results = []
    for req in requests:
        print(f"Processing request with context_id: {req.context_id}")
        result = await base_api.create_chat_completion(req, raw_request)

        # Returning the request_id back since the requests might have been reordered
        results.append({"request_id": req.request_id, "response": json.loads(result.body)}) 
=======
    best_paper = rag_search(request.messages[-1]["content"])
    request.messages.insert(0, {"role": "user", "content": _rag_db[best_paper][1]})
    return await base_api.create_chat_completion(request, raw_request)

@extended_router.post("/chat/completions/batch")
async def create_batch_completion(batch: BatchRequest, raw_request: Request):
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

    # Process sequentially
    results = [None] * len(batch.requests)
    for paper, retrieval_time, req, original_idx in classified:
        req.messages.insert(0, {"role": "user", "content": _rag_db[paper][1]})
        t0 = time.time()
        result = await base_api.create_chat_completion(req, raw_request)
        inference_time = time.time() - t0
        results[original_idx] = {
            "predicted_paper": paper,
            "retrieval_time":  retrieval_time,
            "inference_time":  inference_time,
        }
>>>>>>> Stashed changes

    return results