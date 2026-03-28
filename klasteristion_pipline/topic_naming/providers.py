from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)


def _extract_json_title(text: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        cleaned = text.strip().splitlines()[0][:80]
        return cleaned, "fallback-no-json"
    try:
        obj = json.loads(match.group(0))
        title = str(obj.get("title", "")).strip()
        reason = str(obj.get("reason", "")).strip()
        if not title:
            return "Без названия", "empty-title"
        return title, reason
    except Exception:
        cleaned = text.strip().splitlines()[0][:80]
        return cleaned, "fallback-bad-json"


class BaseNamingProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        text = self.generate_raw(system_prompt, user_prompt)
        return _extract_json_title(text)

    def generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


@dataclass
class YandexGPTProvider(BaseNamingProvider):
    model_name: str = "yandexgpt-lite/latest"
    api_key: str | None = None
    catalog_id: str | None = None
    temperature: float = 0.2
    max_tokens: int = 200

    def generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        key = self.api_key or os.getenv("YANDEX_API_KEY")
        catalog = self.catalog_id or os.getenv("YANDEX_CATALOG_ID")
        if not key:
            raise RuntimeError("YANDEX_API_KEY is not set")
        if not catalog:
            raise RuntimeError("YANDEX_CATALOG_ID is not set")

        model_name = self.model_name
        if model_name in {"yandexgpt", "yandexgpt-lite"}:
            model_name = f"{model_name}/latest"

        payload = {
            "modelUri": f"gpt://{catalog}/{model_name}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }
        resp = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {key}",
            },
            json=payload,
            timeout=90,
        )
        if not resp.ok:
            details = resp.text[:2000]
            raise RuntimeError(
                f"Yandex API request failed (HTTP {resp.status_code}). "
                f"modelUri={payload['modelUri']}. Response: {details}"
            )
        data = resp.json()
        text = (
            data.get("result", {})
            .get("alternatives", [{}])[0]
            .get("message", {})
            .get("text", "")
        )
        return text


@dataclass
class LocalHFProvider(BaseNamingProvider):
    model_name: str
    max_new_tokens: int = 120
    temperature: float = 0.2

    def __post_init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading local topic naming model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        prompt = (
            "<|system|>\n" + system_prompt + "\n"
            "<|user|>\n" + user_prompt + "\n"
            "<|assistant|>\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text


def make_provider(
    model_key: str,
    *,
    yandex_api_key: str | None = None,
    yandex_catalog_id: str | None = None,
):
    if model_key.startswith("yandex:"):
        return YandexGPTProvider(
            model_name=model_key.split(":", 1)[1],
            api_key=yandex_api_key,
            catalog_id=yandex_catalog_id,
        )
    if model_key.startswith("local:"):
        return LocalHFProvider(model_name=model_key.split(":", 1)[1])
    raise ValueError(f"Unknown topic naming model: {model_key}")
