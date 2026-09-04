"""SQLite + SQLAlchemy persistence for module_e (`docs/SUJET.md` MOD-E `database.py`).

Three tables, only the first of which the spec names explicitly (the services catalogue) — `User`
and `Job` are structurally implied by the JWT-scoped auth and the `/optimize/{job_id}` polling
endpoint, so they're defined here too. `MODULE_E_DATABASE_URL` lets tests point at
`sqlite:///:memory:` instead of the on-disk file used at runtime.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./module_e.db"
CATALOG_SEED_PATH = Path(__file__).parent / "catalog_seed.json"


def _database_url() -> str:
    return os.environ.get("MODULE_E_DATABASE_URL", DEFAULT_DATABASE_URL)


def _make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


engine = _make_engine(_database_url())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Service(Base):
    """One entry of the mobile-services catalogue (`docs/SUJET.md` MOD-E, 15-20 seeded rows)."""

    __tablename__ = "services"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)  # Bearer/Teleservice/Supplementary/VAS
    protocol: Mapped[str] = mapped_column(String, nullable=False)
    codec: Mapped[str | None] = mapped_column(String, nullable=True)
    min_mos: Mapped[float] = mapped_column(Float, nullable=False)
    min_bitrate_kbps: Mapped[float] = mapped_column(Float, nullable=False)
    max_delay_ms: Mapped[float] = mapped_column(Float, nullable=False)
    tariff: Mapped[str] = mapped_column(String, nullable=False)  # prepaid / postpaid


class User(Base):
    """Registered API user (`POST /auth/register`); JWT scope is one of admin/operator/user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    scope: Mapped[str] = mapped_column(String, nullable=False, default="user")


class Job(Base):
    """One `/optimize/run` background run, polled via `GET /optimize/{job_id}`."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # uuid4 hex
    problem: Mapped[str] = mapped_column(String, nullable=False)  # pb1_codec / pb2_bts / pb3_qos
    algorithm: Mapped[str] = mapped_column(String, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")  # queued/running/done/error
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_services_if_empty(db: Session, seed_path: Path = CATALOG_SEED_PATH) -> int:
    """Populate `services` from `catalog_seed.json` iff the table is empty. Returns rows inserted."""
    if db.query(Service).first() is not None:
        return 0
    entries = json.loads(seed_path.read_text(encoding="utf-8"))
    db.add_all(Service(**entry) for entry in entries)
    db.commit()
    return len(entries)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_services_if_empty(db)
