"""Database setup. SQLite by default; set DATABASE_URL for PostgreSQL
(e.g. postgresql+psycopg://user:pass@host/gpucalc)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _default_url() -> str:
    data_dir = Path(os.getenv("GPUCALC_DATA_DIR", Path(__file__).resolve().parent.parent / "var"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'gpucalc.db'}"


DATABASE_URL = os.getenv("DATABASE_URL") or _default_url()


def make_engine(url: str):
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kwargs)


engine = make_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ModelRow(Base):
    __tablename__ = "catalog_models"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    family: Mapped[str] = mapped_column(String(100), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_status: Mapped[str] = mapped_column(String(20))
    spec: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class GPURow(Base):
    __tablename__ = "gpus"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    spec: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class VGPUProfileRow(Base):
    __tablename__ = "vgpu_profiles"
    profile_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    physical_gpu: Mapped[str] = mapped_column(String(100), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    spec: Mapped[dict] = mapped_column(JSON)


class InventoryRow(Base):
    __tablename__ = "inventory"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gpu_id: Mapped[str] = mapped_column(String(100), index=True)
    site: Mapped[str] = mapped_column(String(100))
    installed: Mapped[int] = mapped_column(Integer)
    available: Mapped[int] = mapped_column(Integer)


class BenchmarkRow(Base):
    __tablename__ = "benchmarks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(100), index=True)
    gpu_id: Mapped[str] = mapped_column(String(100), index=True)
    precision: Mapped[str] = mapped_column(String(20))
    spec: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SavedCalculationRow(Base):
    __tablename__ = "saved_calculations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    request: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AIFlavorRow(Base):
    __tablename__ = "ai_flavors"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    spec: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class DeploymentSpecRow(Base):
    __tablename__ = "deployment_specs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    workload_type: Mapped[str] = mapped_column(String(40))
    ai_flavor: Mapped[str] = mapped_column(String(100))
    spec: Mapped[dict] = mapped_column(JSON)
    openstack: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SettingRow(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


def init_db(bind=None) -> None:
    Base.metadata.create_all(bind=bind or engine)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
