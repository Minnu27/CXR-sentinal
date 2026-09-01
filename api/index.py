"""Stable ASGI entrypoint used by Vercel's Python runtime.

Keeping this tiny avoids relying on unsupported project metadata to discover the
application. Vercel imports the exported ``app`` object for every invocation.
"""

from src.meditrace.api import app

__all__ = ["app"]
