"""
Qdrant collection management + batch upsert.

Embedded mode: a single on-disk path holds all collections. Only ONE process
can have it open at a time — the Streamlit app keeps an exclusive client.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from rag_pipline.config import (
    EMBEDDING_DIM,
    INDEX_META_DIR,
    QDRANT_COLLECTION_PREFIX,
    QDRANT_PAYLOAD_INDEX_FIELDS,
    QDRANT_STORAGE_DIR,
)

log = logging.getLogger(__name__)


def make_point_id(review_id: str) -> str:
    """Deterministic UUID5 from review_id — idempotent across re-runs."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(review_id)))


def collection_name_for(product_id: int) -> str:
    return f"{QDRANT_COLLECTION_PREFIX}{int(product_id)}"



def _meta_path(name: str) -> Path:
    return INDEX_META_DIR / f"{name}.json"


def write_manifest(name: str, manifest: dict) -> None:
    p = _meta_path(name)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_manifest(name: str) -> dict | None:
    p = _meta_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_manifests() -> list[dict]:
    out = []
    for f in sorted(INDEX_META_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = f.name
            out.append(data)
        except Exception:
            continue
    return out



class QdrantStore:
    """Thin wrapper around qdrant-client with project-specific helpers."""

    def __init__(self, storage_path: str | Path = QDRANT_STORAGE_DIR):
        from qdrant_client import QdrantClient

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        log.info("Qdrant local storage: %s", self.storage_path)
        self.client = QdrantClient(path=str(self.storage_path))


    def list_collections(self) -> list[str]:
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception as exc:
            log.warning("Failed to list collections: %s", exc)
            return []

    def collection_exists(self, name: str) -> bool:
        try:
            return self.client.collection_exists(name)
        except Exception:
            try:
                self.client.get_collection(name)
                return True
            except Exception:
                return False

    def delete_collection(self, name: str) -> None:
        if self.collection_exists(name):
            self.client.delete_collection(name)
        meta = _meta_path(name)
        if meta.exists():
            meta.unlink()

    def recreate_collection(self, name: str, vector_size: int = EMBEDDING_DIM) -> None:
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        if self.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        type_map = {
            "integer": PayloadSchemaType.INTEGER,
            "keyword": PayloadSchemaType.KEYWORD,
            "datetime": PayloadSchemaType.DATETIME,
        }
        for field, kind in QDRANT_PAYLOAD_INDEX_FIELDS.items():
            schema = type_map.get(kind)
            if schema is None:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                log.warning("Failed to create payload index on '%s': %s", field, exc)


    def upsert_points(
        self,
        name: str,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict],
        batch_size: int = 64,
        progress_callback=None,
    ) -> int:
        from qdrant_client.models import PointStruct

        n = len(ids)
        assert len(vectors) == n == len(payloads), "Length mismatch in upsert_points"

        for i in range(0, n, batch_size):
            chunk_ids = ids[i:i + batch_size]
            chunk_vecs = vectors[i:i + batch_size]
            chunk_pl = payloads[i:i + batch_size]
            points = [
                PointStruct(
                    id=cid,
                    vector=v.tolist() if hasattr(v, "tolist") else list(v),
                    payload=p,
                )
                for cid, v, p in zip(chunk_ids, chunk_vecs, chunk_pl)
            ]
            self.client.upsert(collection_name=name, points=points)
            if progress_callback is not None:
                try:
                    progress_callback(min(i + batch_size, n), n)
                except Exception:
                    pass
        return n


    def search(
        self,
        name: str,
        query_vector: np.ndarray,
        top_k: int,
        qdrant_filter=None,
    ):
        """
        Top-K nearest neighbours.

        Uses `query_points` (qdrant-client ≥ 1.10), falls back to deprecated
        `search` for older clients. Always returns a list of ScoredPoint-like
        objects with `.score` and `.payload`.
        """
        vec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=name,
                query=vec,
                query_filter=qdrant_filter,
                limit=int(top_k),
                with_payload=True,
                with_vectors=False,
            )
            return getattr(resp, "points", resp)
        return self.client.search(
            collection_name=name,
            query_vector=vec,
            query_filter=qdrant_filter,
            limit=int(top_k),
            with_payload=True,
            with_vectors=False,
        )


    def collection_info(self, name: str) -> dict:
        try:
            info = self.client.get_collection(name)
            return {
                "points_count": getattr(info, "points_count", None),
                "vectors_count": getattr(info, "vectors_count", None),
                "status": str(getattr(info, "status", "")),
                "config": str(getattr(info, "config", "")),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def scroll_all(self, name: str, limit: int = 10000) -> list:
        offset = None
        out = []
        while True:
            try:
                points, offset = self.client.scroll(
                    collection_name=name,
                    limit=min(256, limit - len(out)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                log.warning("scroll failed: %s", exc)
                break
            out.extend(points)
            if offset is None or len(out) >= limit:
                break
        return out



def build_manifest(
    *,
    collection_name: str,
    product_id: int,
    n_records: int,
    embedding_model: str,
    sentiment_model: str,
    csv_file: str,
    sentiment_dist: dict,
    gender_dist: dict,
    rating_dist: dict,
    elapsed_sec: float,
) -> dict:
    return {
        "collection": collection_name,
        "product_id": product_id,
        "n_records": n_records,
        "csv_file": csv_file,
        "embedding_model": embedding_model,
        "sentiment_model": sentiment_model,
        "sentiment_distribution": sentiment_dist,
        "gender_distribution": gender_dist,
        "rating_distribution": rating_dist,
        "indexed_at": datetime.utcnow().isoformat() + "Z",
        "elapsed_sec": round(elapsed_sec, 2),
    }
