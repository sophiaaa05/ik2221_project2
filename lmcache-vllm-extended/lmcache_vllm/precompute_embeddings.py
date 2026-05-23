"""
Run once before any experiments to pre-compute paper embeddings.
Output: rag_index.pt
Usage: python precompute_embeddings.py
"""

import os
import torch
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DATA_DIR         = Path(__file__).resolve().parent.parent / "frontend" / "data"
INDEX_PATH       = Path(__file__).resolve().parent.parent / "frontend" / "rag_index.pt"

def build_index(data_dir, model):
    """
    Embed all papers in data_dir and return the full index.
    Always embeds all papers — the server slices to N_DOCS at startup.
    """
    index = {}
    for filename in sorted(f for f in os.listdir(data_dir) if f.endswith(".txt")):
        key = filename.removesuffix(".txt")
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
        embedding = model.encode(text, convert_to_tensor=True).unsqueeze(0).cpu()
        index[key] = (embedding, text)
        print(f"[PRECOMPUTE] Embedded: {key}")
    return index

if __name__ == "__main__":
    print("[PRECOMPUTE] Loading model...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
    print(f"[PRECOMPUTE] Building index from {DATA_DIR}...")
    index = build_index(DATA_DIR, model)
    torch.save(index, INDEX_PATH)
    print(f"[PRECOMPUTE] Saved {len(index)} embeddings to {INDEX_PATH}")