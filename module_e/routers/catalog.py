"""`GET /services` (filterable) + `GET /services/{id}` (`docs/SUJET.md` MOD-E)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from module_e.auth import get_current_user
from module_e.database import Service, get_db
from module_e.models import ServiceOut
from module_e.rate_limit import RATE_LIMIT, limiter

router = APIRouter(prefix="/services", tags=["catalog"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[ServiceOut])
@limiter.limit(RATE_LIMIT)
def list_services(
    request: Request,
    category: str | None = None,
    codec: str | None = None,
    min_mos: float | None = None,
    db: Session = Depends(get_db),
) -> list[Service]:
    query = db.query(Service)
    if category is not None:
        query = query.filter(Service.category == category)
    if codec is not None:
        query = query.filter(Service.codec == codec)
    if min_mos is not None:
        query = query.filter(Service.min_mos >= min_mos)
    return query.order_by(Service.id).all()


@router.get("/{service_id}", response_model=ServiceOut)
@limiter.limit(RATE_LIMIT)
def get_service(request: Request, service_id: int, db: Session = Depends(get_db)) -> Service:
    service = db.get(Service, service_id)
    if service is None:
        raise HTTPException(404, detail=f"Service {service_id} not found")
    return service
