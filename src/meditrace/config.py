from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./meditrace.db"
    object_store_path: str = "./data/documents"
    max_upload_bytes: int = 20 * 1024 * 1024
    model_endpoint: str | None = None
    model_api_key: str | None = None
    model_name: str = "medgemma"
    auto_process: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        vercel = os.getenv("VERCEL") == "1"
        database_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if database_url and database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        return cls(
            database_url=database_url
            or ("sqlite:////tmp/meditrace.db" if vercel else cls.database_url),
            object_store_path=os.getenv("OBJECT_STORE_PATH")
            or ("/tmp/meditrace-documents" if vercel else cls.object_store_path),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
            model_endpoint=os.getenv("MODEL_ENDPOINT") or None,
            model_api_key=os.getenv("MODEL_API_KEY") or None,
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            auto_process=os.getenv("AUTO_PROCESS", "false").lower()
            in {"1", "true", "yes"},
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
