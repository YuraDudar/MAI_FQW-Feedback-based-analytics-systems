import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.services.job_service import JobService
from backend.models.db_models import JobStatus


async def _setup_user_product(db: AsyncSession, suffix: str):
    from backend.core.security import hash_password
    await db.execute(text("""
        INSERT OR IGNORE INTO data_sources (source_id, name, platform, site_url)
        VALUES (1, 'Wildberries', 'wildberries', 'https://www.wildberries.ru')
    """))
    await db.execute(text("""
        INSERT INTO users (username, email, password_hash, role)
        VALUES (:u, :e, :h, 'analyst')
    """), {"u": f"user_{suffix}", "e": f"{suffix}@t.com", "h": hash_password("pass")})
    await db.flush()
    user_id = (await db.execute(text(
        "SELECT user_id FROM users WHERE username = :u"
    ), {"u": f"user_{suffix}"})).scalar_one()

    await db.execute(text("""
        INSERT INTO products (name, source_product_id, source_id, user_id)
        VALUES ('Test Product', :sku, 1, :uid)
    """), {"sku": f"sku_{suffix}", "uid": user_id})
    await db.flush()
    product_id = (await db.execute(text(
        "SELECT product_id FROM products WHERE user_id = :uid ORDER BY product_id DESC LIMIT 1"
    ), {"uid": user_id})).scalar_one()
    return user_id, product_id


def _make_kafka_mock():
    kafka = MagicMock()
    kafka.send = AsyncMock()
    return kafka


def _make_redis_mock():
    redis = MagicMock()
    redis.set_product_status = AsyncMock()
    redis.invalidate_dashboard = AsyncMock()
    return redis


@pytest.mark.asyncio
async def test_create_parse_job(db_session: AsyncSession):
    user_id, product_id = await _setup_user_product(db_session, "jb1")
    kafka = _make_kafka_mock()
    redis = _make_redis_mock()

    svc = JobService(db_session, kafka, redis)
    job = await svc.create_parse_job(product_id, user_id)

    assert job.job_id is not None
    assert job.job_type.value == "parsing"
    assert job.status.value == "pending"
    kafka.send.assert_called_once()
    redis.set_product_status.assert_called_once_with(product_id, "parsing")


@pytest.mark.asyncio
async def test_get_job(db_session: AsyncSession):
    user_id, product_id = await _setup_user_product(db_session, "jb2")
    kafka = _make_kafka_mock()
    redis = _make_redis_mock()
    svc = JobService(db_session, kafka, redis)
    created = await svc.create_parse_job(product_id, user_id)

    fetched = await svc.get_job(created.job_id, user_id)
    assert fetched.job_id == created.job_id


@pytest.mark.asyncio
async def test_get_job_wrong_user_raises(db_session: AsyncSession):
    from fastapi import HTTPException
    from backend.core.security import hash_password

    user_id, product_id = await _setup_user_product(db_session, "jb3")
    await db_session.execute(text("""
        INSERT INTO users (username, email, password_hash, role)
        VALUES ('other_jb3', 'other_jb3@t.com', :h, 'analyst')
    """), {"h": hash_password("pass")})
    await db_session.flush()
    other_id = (await db_session.execute(text(
        "SELECT user_id FROM users WHERE username = 'other_jb3'"
    ))).scalar_one()

    kafka = _make_kafka_mock()
    redis = _make_redis_mock()
    svc = JobService(db_session, kafka, redis)
    job = await svc.create_parse_job(product_id, user_id)

    with pytest.raises(HTTPException) as exc:
        await svc.get_job(job.job_id, other_id)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_jobs(db_session: AsyncSession):
    user_id, product_id = await _setup_user_product(db_session, "jb4")
    kafka = _make_kafka_mock()
    redis = _make_redis_mock()
    svc = JobService(db_session, kafka, redis)
    await svc.create_parse_job(product_id, user_id)
    await svc.create_parse_job(product_id, user_id)

    jobs = await svc.list_jobs(user_id)
    assert len(jobs) >= 2


@pytest.mark.asyncio
async def test_update_job_status(db_session: AsyncSession):
    user_id, product_id = await _setup_user_product(db_session, "jb5")
    kafka = _make_kafka_mock()
    redis = _make_redis_mock()
    svc = JobService(db_session, kafka, redis)
    job = await svc.create_parse_job(product_id, user_id)

    await svc.update_job_status(job.job_id, JobStatus.running)
    fetched = await svc.get_job(job.job_id, user_id)
    assert fetched.status.value == "running"
