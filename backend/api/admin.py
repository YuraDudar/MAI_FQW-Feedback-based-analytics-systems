import logging

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_admin_id
from models.db_models import AnalysisJob, JobStatus
from models.schemas import AdminJobStats

import sys
sys.path.insert(0, "/app")
from infrastructure.config import ML_SERVICE_INTERNAL_URL

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Администратор"])


@router.get("/jobs/stats", response_model=AdminJobStats)
async def job_stats(
    admin_id: int = Depends(get_current_admin_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AnalysisJob.status, func.count().label("cnt"))
        .group_by(AnalysisJob.status)
    )
    rows = result.all()
    counts = {r.status.value: r.cnt for r in rows}
    return AdminJobStats(
        pending=counts.get("pending", 0),
        running=counts.get("running", 0),
        completed=counts.get("completed", 0),
        failed=counts.get("failed", 0),
        total=sum(counts.values()),
    )


@router.get("/ml/health")
async def ml_health(admin_id: int = Depends(get_current_admin_id)):
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{ML_SERVICE_INTERNAL_URL}/health")
            return {"status": resp.json().get("status"), "code": resp.status_code}
        except Exception as exc:
            return {"status": "недоступен", "error": str(exc)}


@router.get("/users/list")
async def list_users(
    admin_id: int = Depends(get_current_admin_id),
    db: AsyncSession = Depends(get_db),
):
    from models.db_models import User
    result = await db.execute(
        select(User).order_by(User.created_at.desc()).limit(200)
    )
    users = result.scalars().all()
    return [
        {
            "user_id": u.user_id,
            "username": u.username,
            "email": u.email,
            "role": u.role.value,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@router.patch("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    active: bool,
    admin_id: int = Depends(get_current_admin_id),
    db: AsyncSession = Depends(get_db),
):
    from models.db_models import User
    from sqlalchemy import update
    await db.execute(
        update(User).where(User.user_id == user_id).values(is_active=active)
    )
    await db.commit()
    return {"ok": True}
