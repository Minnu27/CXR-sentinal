from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./meditrace.db"
    object_store_path: str = "./data/documents"
    max_upload_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            object_store_path=os.getenv("OBJECT_STORE_PATH", cls.object_store_path),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", cls.max_upload_bytes)),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
