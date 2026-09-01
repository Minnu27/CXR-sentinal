from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./meditrace.db"
    object_store_path: str = "./data/documents"
    max_upload_bytes: int = 20 * 1024 * 1024

    @staticmethod
    def _database_url() -> str:
        url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if url:
            # Vercel/Neon commonly expose a generic Postgres URL. Explicitly
            # select psycopg 3, which is the driver shipped with this app.
            if url.startswith("postgres://"):
                return url.replace("postgres://", "postgresql+psycopg://", 1)
            if url.startswith("postgresql://"):
                return url.replace("postgresql://", "postgresql+psycopg://", 1)
            return url
        if os.getenv("VERCEL"):
            return "sqlite:////tmp/meditrace.db"
        return Settings.database_url

    @staticmethod
    def _object_store_path() -> str:
        configured_path = os.getenv("OBJECT_STORE_PATH")
        if configured_path:
            return configured_path
        # A Vercel function's application directory is read-only. /tmp is the
        # only writable location and is intentionally treated as demo-only,
        # ephemeral storage until an S3 adapter is configured.
        if os.getenv("VERCEL"):
            return "/tmp/meditrace-documents"
        return Settings.object_store_path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=cls._database_url(),
            object_store_path=cls._object_store_path(),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
