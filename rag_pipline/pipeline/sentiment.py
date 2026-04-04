"""
Russian sentiment classifier with strict GPU lifecycle management.

State machine:
  IDLE → load() → LOADED → predict() → ... → unload() → IDLE

Outputs are normalised to {"positive", "negative", "neutral"}.
Different HF checkpoints use different label conventions; we normalise via
substring match on `id2label` so swapping models is transparent.
"""
from __future__ import annotations

import gc
import logging
from typing import Iterable

from rag_pipline.config import (
    DEFAULT_SENTIMENT_MODEL,
    SENTIMENT_LABELS,
    SENTIMENT_MODELS,
)

log = logging.getLogger(__name__)


def _normalize_label(raw: str) -> str:
    """Map model-specific labels to {positive, negative, neutral}."""
    s = str(raw).lower()
    if "pos" in s or s in ("1", "label_2", "positive"):
        if "pos" in s:
            return "positive"
    if "neg" in s:
        return "negative"
    if "neu" in s:
        return "neutral"
    if "pos" in s:
        return "positive"
    if s.isdigit():
        idx = int(s)
        if idx == 0:
            return "negative"
        if idx == 1:
            return "neutral"
        if idx == 2:
            return "positive"
    return "neutral"


class SentimentAnalyzer:
    def __init__(self, model_key: str = DEFAULT_SENTIMENT_MODEL, device: str = "auto"):
        if model_key not in SENTIMENT_MODELS:
            raise ValueError(f"Unknown sentiment model '{model_key}'. Choose from {list(SENTIMENT_MODELS)}")
        self.model_key = model_key
        self.cfg = SENTIMENT_MODELS[model_key]
        self._device = device  
        self._model = None
        self._tokenizer = None
        self._id2label: dict[int, str] = {}


    def _resolve_device(self) -> str:
        import torch
        if self._device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = self.cfg["name"]
        device = self._resolve_device()
        log.info("Loading sentiment model: %s on %s", name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        self._model.eval()
        self._device_resolved = device
        try:
            id2label = dict(self._model.config.id2label)  
            self._id2label = {int(k): str(v) for k, v in id2label.items()}
        except Exception:
            self._id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


    def predict(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
        progress_callback=None,
    ) -> list[str]:
        """
        Returns a list of normalised labels — same length as `texts`.
        progress_callback(i, total) is called after each batch (for Streamlit progress).
        """
        if self._model is None:
            self.load()

        import torch

        bs = batch_size or self.cfg["batch_size"]
        ml = max_length or self.cfg["max_seq_length"]
        results: list[str] = []
        total = len(texts)

        for i in range(0, total, bs):
            batch = texts[i:i + bs]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=ml,
                return_tensors="pt",
            ).to(self._device_resolved)
            with torch.no_grad():
                logits = self._model(**enc).logits
            preds = logits.argmax(-1).detach().cpu().tolist()
            for p in preds:
                raw_label = self._id2label.get(int(p), str(p))
                results.append(_normalize_label(raw_label))
            if progress_callback is not None:
                try:
                    progress_callback(min(i + bs, total), total)
                except Exception:
                    pass

        return results


    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.unload()


def label_distribution(labels: Iterable[str]) -> dict[str, int]:
    """Count labels (always returns all 3 keys)."""
    out = {k: 0 for k in SENTIMENT_LABELS}
    for l in labels:
        if l in out:
            out[l] += 1
        else:
            out["neutral"] += 1
    return out
