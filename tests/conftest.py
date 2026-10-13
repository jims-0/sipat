import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def app(tmp_path):
    from app import create_app

    app = create_app({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}",
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
    })
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def jpeg_bytes(color=(60, 140, 60), size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()
