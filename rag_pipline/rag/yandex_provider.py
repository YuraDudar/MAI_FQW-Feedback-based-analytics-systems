"""
YandexGPT provider via the official `yandex-cloud-ml-sdk`, with REST fallback.

Why a fallback? The SDK is the recommended path (mandated by the project spec)
but environments without it should still be debuggable — REST keeps parity with
the existing klasteristion_pipline behaviour.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from rag_pipline.config import (
    YANDEX_LITE_MAX_TOKENS,
    YANDEX_LITE_MODEL,
    YANDEX_LITE_TEMPERATURE,
    YANDEX_MODEL_VERSION,
    YANDEX_PRO_MAX_TOKENS,
    YANDEX_PRO_MODEL,
    YANDEX_PRO_TEMPERATURE,
)

log = logging.getLogger(__name__)


class YandexConfigError(RuntimeError):
    """Raised when required Yandex credentials are missing."""


@dataclass
class YandexLLM:
    """Single Yandex model handle.

    Pick a `kind`:
        kind='lite' → query expansion (yandexgpt-lite, low temperature, short)
        kind='pro'  → answer generation (yandexgpt, longer outputs)
    """
    folder_id: str
    api_key: str
    kind: str = "lite"   
    model_version: str = YANDEX_MODEL_VERSION
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_sec: float = 60.0
    use_sdk: bool = True

    _sdk_model: object = field(default=None, init=False, repr=False)
    _sdk_available: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not self.folder_id:
            raise YandexConfigError("Yandex folder_id (catalog_id) is required")
        if not self.api_key:
            raise YandexConfigError("Yandex API key is required")

        if self.kind in ("lite", YANDEX_LITE_MODEL):
            self._model_id = YANDEX_LITE_MODEL
            if self.temperature is None:
                self.temperature = YANDEX_LITE_TEMPERATURE
            if self.max_tokens is None:
                self.max_tokens = YANDEX_LITE_MAX_TOKENS
        elif self.kind in ("pro", YANDEX_PRO_MODEL):
            self._model_id = YANDEX_PRO_MODEL
            if self.temperature is None:
                self.temperature = YANDEX_PRO_TEMPERATURE
            if self.max_tokens is None:
                self.max_tokens = YANDEX_PRO_MAX_TOKENS
        else:
            self._model_id = self.kind
            if self.temperature is None:
                self.temperature = YANDEX_LITE_TEMPERATURE
            if self.max_tokens is None:
                self.max_tokens = YANDEX_LITE_MAX_TOKENS

        if self.use_sdk:
            self._init_sdk()


    def _init_sdk(self) -> None:
        try:
            from yandex_cloud_ml_sdk import YCloudML
        except ImportError:
            log.warning("yandex-cloud-ml-sdk not installed — falling back to REST")
            self._sdk_available = False
            return

        try:
            sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
            self._sdk_model = sdk.models.completions(
                self._model_id,
                model_version=self.model_version,
            )
            self._sdk_available = True
        except Exception as exc:
            log.warning("Yandex SDK init failed (%s) — will use REST", exc)
            self._sdk_available = False

    def _sdk_generate(self, system_prompt: str, user_prompt: str, *, temperature: float, max_tokens: int) -> str:
        configured = self._sdk_model.configure(temperature=temperature, max_tokens=max_tokens)
        result = configured.run([
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt},
        ])
        alternatives = getattr(result, "alternatives", None)
        if alternatives:
            return str(alternatives[0].text)
        try:
            return str(result[0].text)
        except Exception:
            pass
        return str(result)


    def _rest_generate(self, system_prompt: str, user_prompt: str, *, temperature: float, max_tokens: int) -> str:
        import requests
        model_name = self._model_id
        if "/" not in model_name:
            model_name = f"{model_name}/{self.model_version}"
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{model_name}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
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
                "Authorization": f"Api-Key {self.api_key}",
            },
            json=payload,
            timeout=self.timeout_sec,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Yandex API failed (HTTP {resp.status_code}). "
                f"modelUri={payload['modelUri']}. Response: {resp.text[:1500]}"
            )
        data = resp.json()
        try:
            return data["result"]["alternatives"][0]["message"]["text"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected Yandex response shape: {json.dumps(data)[:800]}") from exc


    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        mtok = self.max_tokens if max_tokens is None else max_tokens

        if self.use_sdk and self._sdk_available:
            try:
                return self._sdk_generate(system_prompt, user_prompt, temperature=temp, max_tokens=mtok)
            except Exception as exc:
                log.warning("SDK call failed (%s) — retrying via REST", exc)
        return self._rest_generate(system_prompt, user_prompt, temperature=temp, max_tokens=mtok)

    @property
    def transport(self) -> str:
        return "sdk" if (self.use_sdk and self._sdk_available) else "rest"

    @property
    def model_id(self) -> str:
        return self._model_id


def build_lite_and_pro(folder_id: str, api_key: str, *, prefer_sdk: bool = True) -> tuple[YandexLLM, YandexLLM]:
    """Convenience: instantiate the two Yandex handles used by the orchestrator."""
    lite = YandexLLM(folder_id=folder_id, api_key=api_key, kind="lite", use_sdk=prefer_sdk)
    pro = YandexLLM(folder_id=folder_id, api_key=api_key, kind="pro", use_sdk=prefer_sdk)
    return lite, pro
