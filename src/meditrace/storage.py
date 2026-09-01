from __future__ import annotations

from pathlib import Path
import os


class LocalObjectStore:
    """Filesystem-backed object store for local work; the API is S3-portable."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, content: bytes) -> None:
        destination = (self.root / key).resolve()
        if self.root not in destination.parents:
            raise ValueError("invalid object key")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(content)
        os.replace(temporary, destination)

    def get(self, key: str) -> bytes:
        source = (self.root / key).resolve()
        if self.root not in source.parents:
            raise ValueError("invalid object key")
        return source.read_bytes()

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)
