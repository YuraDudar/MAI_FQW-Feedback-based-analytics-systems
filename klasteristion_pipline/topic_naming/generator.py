from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from .prompts import SYSTEM_PROMPT, build_batch_user_prompt, build_user_prompt
from .providers import make_provider


def build_topic_payloads(
    topics: list[dict],
    topic_documents: dict[int, list[dict]],
    *,
    random_seed: int = 42,
    sample_size: int = 10,
) -> list[dict]:
    rng = random.Random(random_seed)
    payloads = []
    for t in topics:
        tid = int(t["topic_id"])
        docs = topic_documents.get(tid, [])
        sampled = docs if len(docs) <= sample_size else rng.sample(docs, sample_size)
        payloads.append(
            {
                "topic_id": tid,
                "keywords": t.get("top_words", [])[:10],
                "keyword_label": t.get("keyword_label", ""),
                "sample_reviews": [str(d.get("text", ""))[:500] for d in sampled],
            }
        )
    return payloads


@dataclass
class TopicTitleGenerator:
    model_key: str
    yandex_api_key: str | None = None
    yandex_catalog_id: str | None = None

    def _parse_batch_output(self, text: str) -> dict[int, dict]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return {}
        rows = obj.get("titles", [])
        out: dict[int, dict] = {}
        for row in rows:
            try:
                tid = int(row.get("topic_id"))
            except Exception:
                continue
            title = str(row.get("title", "")).strip()
            reason = str(row.get("reason", "")).strip()
            if title:
                out[tid] = {"title": title[:90], "reason": reason}
        return out

    def generate_for_pool(
        self,
        pool_name: str,
        topic_payloads: list[dict],
        *,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> dict[int, dict]:
        provider = make_provider(
            self.model_key,
            yandex_api_key=self.yandex_api_key,
            yandex_catalog_id=self.yandex_catalog_id,
        )
        user_prompt = build_batch_user_prompt(pool_name, topic_payloads)
        raw_text = provider.generate_raw(system_prompt, user_prompt)
        parsed = self._parse_batch_output(raw_text)
        result: dict[int, dict] = {}
        parse_reason = "ok" if parsed else "batch-parse-failed"

        for payload in topic_payloads:
            tid = int(payload["topic_id"])
            if tid in parsed:
                result[tid] = parsed[tid]

        used_titles = [v.get("title", "") for v in result.values() if v.get("title")]
        for payload in topic_payloads:
            tid = int(payload["topic_id"])
            if tid in result:
                continue
            single_prompt = build_user_prompt(
                pool_name=pool_name,
                topic_id=tid,
                keywords=payload.get("keywords", []),
                used_titles=used_titles,
                sample_reviews=payload.get("sample_reviews", []),
            )
            try:
                title, reason = provider.generate(system_prompt, single_prompt)
                title = title.strip()[:90] if title else "Без названия"
                used_titles.append(title)
                result[tid] = {
                    "title": title,
                    "reason": f"fallback-sequential:{reason or parse_reason}",
                }
            except Exception as exc:
                result[tid] = {
                    "title": "Без названия",
                    "reason": f"{parse_reason}; fallback-error:{exc}",
                }
        return result
