"""Compare the running container with the in-process app.
Run: docker run -p 8000:8000 covid-api  then  API_URL=http://localhost:8000 pytest tests/test_local_vs_container.py
"""
import os

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.test_api import SAMPLE

API_URL = os.getenv("API_URL")


@pytest.mark.skipif(not API_URL, reason="set API_URL to test a running container")
def test_container_matches_local():
    import httpx
    remote = httpx.post(f"{API_URL}/predict", json=SAMPLE, timeout=10).json()
    local = TestClient(app).post("/predict", json=SAMPLE).json()
    assert remote == local
