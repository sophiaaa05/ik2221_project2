"""
Run this script once before starting the server to pre-compute paper embeddings.
Output: rag_index.pt in the project data directory.
Usage: 
python lmcache-vllm-extended/lmcache_vllm/precompute_embeddings.py 
"""

import os
import torch
import argparse  
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DATA_DIR         = Path(__file__).resolve().parent.parent / "frontend" / "data"
INDEX_PATH       = Path(__file__).resolve().parent.parent / "frontend" / "rag_index.pt"

def build_index(data_dir, model, n_docs=None):
    index = {}
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if n_docs is not None:
        files = files[:n_docs]
    for filename in files:
        key = filename.removesuffix(".txt")
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        embedding = model.encode(text, convert_to_tensor=True).unsqueeze(0).cpu()
        index[key] = (embedding, text)
        print(f"[PRECOMPUTE] Embedded: {key}")
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=None,
                        help="Number of documents to embed (default: all)")
    args = parser.parse_args()

    print(f"[PRECOMPUTE] Loading model...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
    print(f"[PRECOMPUTE] Building index from {DATA_DIR}...")
    index = build_index(DATA_DIR, model, n_docs=args.n_docs)
    torch.save(index, INDEX_PATH)
    print(f"[PRECOMPUTE] Saved {len(index)} embeddings to {INDEX_PATH}")