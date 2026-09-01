from src.meditrace.config import Settings


def test_vercel_asgi_entrypoint_imports(monkeypatch, tmp_path):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'vercel.db'}")
    monkeypatch.setenv("OBJECT_STORE_PATH", str(tmp_path / "objects"))

    from api.index import app

    assert app.title == "MediTrace AI"


def test_vercel_defaults_use_writable_tmp(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("OBJECT_STORE_PATH", raising=False)

    settings = Settings.from_env()

    assert settings.database_url == "sqlite:////tmp/meditrace.db"
    assert settings.object_store_path == "/tmp/meditrace-documents"


def test_vercel_postgres_url_selects_installed_psycopg_driver(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://user:secret@example.test/meditrace")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings.from_env()

    assert settings.database_url == "postgresql+psycopg://user:secret@example.test/meditrace"


def test_explicit_configuration_wins_on_vercel(monkeypatch, tmp_path):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("DATABASE_URL", "sqlite:////tmp/custom.db")
    monkeypatch.setenv("OBJECT_STORE_PATH", str(tmp_path))

    settings = Settings.from_env()

    assert settings.database_url == "sqlite:////tmp/custom.db"
    assert settings.object_store_path == str(tmp_path)
