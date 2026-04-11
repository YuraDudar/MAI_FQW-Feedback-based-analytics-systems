import asyncio
import logging
import sys
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/klasteristion_pipline")
sys.path.insert(0, "/app/rag_pipline")

from infrastructure.config import (
    KAFKA_TOPIC_CLUSTER_JOBS,
    KAFKA_TOPIC_ANALYSIS_DONE,
    QDRANT_HOST, QDRANT_PORT,
    QDRANT_COLLECTION_PREFIX,
    QDRANT_VECTOR_DIM,
)
from models.db_models import Cluster, ReviewClusterMapping, ReviewNLP, ReviewSentiment

logger = logging.getLogger(__name__)


class ClusteringService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def run_full_pipeline(self, product_id: int, job_id: int, reviews: list[dict]):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_pipeline_sync, product_id, job_id, reviews
        )
        await self._save_results(product_id, job_id, result)
        await self._upsert_qdrant(product_id, reviews, result)
        return result

    def _run_pipeline_sync(self, product_id: int, job_id: int, reviews: list[dict]) -> dict:
        try:
            from klasteristion_pipline.pipeline.preprocessor import Preprocessor
            from klasteristion_pipline.pipeline.embedder import Embedder
            from klasteristion_pipline.pipeline.clusterer import Clusterer
            from klasteristion_pipline.topic_naming.generator import TopicNameGenerator
            from klasteristion_pipline.config import DEFAULT_EMBEDDING_MODEL

            preprocessor = Preprocessor()
            pos_texts, neg_texts, pos_ids, neg_ids = preprocessor.split_pools(reviews)

            embedder = Embedder(DEFAULT_EMBEDDING_MODEL)
            pos_embeddings = embedder.embed(pos_texts) if pos_texts else None
            neg_embeddings = embedder.embed(neg_texts) if neg_texts else None

            clusterer = Clusterer()
            pos_clusters = clusterer.cluster(pos_embeddings, pos_texts) if pos_embeddings is not None else []
            neg_clusters = clusterer.cluster(neg_embeddings, neg_texts) if neg_embeddings is not None else []

            namer = TopicNameGenerator()
            named_pos = namer.name_topics(pos_clusters)
            named_neg = namer.name_topics(neg_clusters)

            return {
                "positive": {"clusters": named_pos, "review_ids": pos_ids},
                "negative": {"clusters": named_neg, "review_ids": neg_ids},
                "pos_count": len(pos_texts),
                "neg_count": len(neg_texts),
            }
        except Exception as exc:
            logger.exception("Ошибка в пайплайне кластеризации: %s", exc)
            raise

    async def _save_results(self, product_id: int, job_id: int, result: dict):
        await self.db.execute(
            delete(Cluster).where(
                Cluster.product_id == product_id,
                Cluster.clustering_job_id == job_id,
            )
        )
        await self.db.flush()

        for sentiment_cat, cat_data in [("positive", result["positive"]), ("negative", result["negative"])]:
            clusters = cat_data.get("clusters", [])
            review_ids = cat_data.get("review_ids", [])

            for topic_data in clusters:
                topic_id = topic_data.get("topic_id", -1)
                cluster = Cluster(
                    clustering_job_id=job_id,
                    product_id=product_id,
                    sentiment_category=sentiment_cat,
                    bertopic_topic_id=topic_id,
                    llm_label=topic_data.get("label"),
                    keywords=topic_data.get("keywords", []),
                    review_count=len(topic_data.get("review_ids", [])),
                    avg_rating=topic_data.get("avg_rating"),
                )
                self.db.add(cluster)
                await self.db.flush()
                await self.db.refresh(cluster)

                for i, rid in enumerate(topic_data.get("review_ids", [])):
                    if rid < len(review_ids):
                        mapping = ReviewClusterMapping(
                            review_id=review_ids[rid] if isinstance(rid, int) else rid,
                            cluster_id=cluster.cluster_id,
                            product_id=product_id,
                            probability=topic_data.get("probabilities", [1.0])[i] if i < len(topic_data.get("probabilities", [])) else 1.0,
                            is_outlier=(topic_id == -1),
                        )
                        self.db.add(mapping)

        await self.db.flush()

    async def _upsert_qdrant(self, product_id: int, reviews: list[dict], result: dict):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct, VectorParams, Distance
            from rag_pipline.pipeline.embedder import E5Embedder
            from rag_pipline.pipeline.indexer import QdrantStore

            store = QdrantStore(host=QDRANT_HOST, port=QDRANT_PORT)
            collection = f"{QDRANT_COLLECTION_PREFIX}{product_id}"
            store.ensure_collection(collection, dim=QDRANT_VECTOR_DIM)

            embedder = E5Embedder()
            review_map = {r["review_id"]: r for r in reviews}

            batch_size = 32
            points = []
            review_list = list(review_map.values())

            for i in range(0, len(review_list), batch_size):
                batch = review_list[i:i + batch_size]
                texts = [
                    " ".join(filter(None, [
                        r.get("advantages", ""),
                        r.get("disadvantages", ""),
                        r.get("comment", ""),
                    ]))
                    for r in batch
                ]
                vectors = embedder.embed_batch(texts)

                for j, review in enumerate(batch):
                    sentiment_label = "neutral"
                    from sqlalchemy import select as sa_select
                    sent_result = await self.db.execute(
                        sa_select(ReviewSentiment.sentiment_label, ReviewSentiment.reviewer_gender)
                        .where(ReviewSentiment.review_id == review["review_id"])
                    )
                    sent_row = sent_result.one_or_none()
                    if sent_row:
                        sentiment_label = sent_row.sentiment_label.value
                        reviewer_gender = sent_row.reviewer_gender or "unknown"
                    else:
                        reviewer_gender = "unknown"

                    points.append(PointStruct(
                        id=abs(hash(review["review_id"])) % (2**63),
                        vector=vectors[j].tolist(),
                        payload={
                            "review_id": review["review_id"],
                            "product_id": product_id,
                            "created_date": str(review.get("created_date", "")),
                            "rating": review.get("rating"),
                            "sentiment_label": sentiment_label,
                            "reviewer_gender": reviewer_gender,
                        },
                    ))

            store.upsert(collection, points)
            logger.info("Upsert %d точек в Qdrant коллекцию %s", len(points), collection)
        except Exception as exc:
            logger.error("Ошибка upsert в Qdrant: %s", exc)
