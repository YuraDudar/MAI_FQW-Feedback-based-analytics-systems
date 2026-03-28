"""
Pre-download all embedding models to the HuggingFace cache.

Run this ONCE before experiments so that model downloads
don't interfere with pipeline execution.

Usage:
    python download_models.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMBEDDING_MODELS, TOPIC_NAMING_MODELS


def _snapshot_exists(repo_id: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def main():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("  PRE-DOWNLOADING MODELS")
    print("=" * 60)

    for key, cfg in EMBEDDING_MODELS.items():
        name = cfg["name"]
        print(f"\n{'─' * 60}")
        print(f"  [{key}] {name}")
        print(f"{'─' * 60}")

        if _snapshot_exists(name):
            print("  ✓ SKIP — already in local cache")
            continue

        t0 = time.perf_counter()
        try:
            model = SentenceTransformer(name, trust_remote_code=True, device="cpu")
            test = model.encode(["тестовое предложение для проверки"])
            dim = test.shape[-1]
            del model
            elapsed = time.perf_counter() - t0
            print(f"  ✓ OK — dim={dim}, loaded in {elapsed:.1f}s")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'=' * 60}")
    print("  PRE-DOWNLOADING TOPIC NAMING MODELS")
    print("=" * 60)

    local_models = {
        k: v.split(":", 1)[1]
        for k, v in TOPIC_NAMING_MODELS.items()
        if v.startswith("local:")
    }

    for key, model_name in local_models.items():
        print(f"\n{'─' * 60}")
        print(f"  [{key}] {model_name}")
        print(f"{'─' * 60}")
        if _snapshot_exists(model_name):
            print("  ✓ SKIP — already in local cache")
            continue
        t0 = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            _ = tokenizer("проверка загрузки", return_tensors="pt")
            del model
            del tokenizer
            elapsed = time.perf_counter() - t0
            print(f"  ✓ OK — loaded in {elapsed:.1f}s")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'=' * 60}")
    print("  ALL MODEL DOWNLOADS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
