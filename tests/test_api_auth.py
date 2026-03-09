import pytest
from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)


@pytest.fixture(autouse=True)
def set_api_tokens_env():
    os.environ["API_TOKENS"] = "test_token_1,test_token_2"
    # Ensure we're NOT in dev mode for auth tests
    os.environ.pop("ENV", None)
    yield
    del os.environ["API_TOKENS"]


def test_classify_unauthorized():
    response = client.post(
        "/api/classify", json={"text": "test", "enabled_models": ["gemini"]}
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "Not authenticated"}


def test_classify_invalid_token():
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.post(
        "/api/classify",
        headers=headers,
        json={"text": "test", "enabled_models": ["gemini"]},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials"}


def test_classify_valid_token():
    headers = {"Authorization": "Bearer test_token_1"}
    response = client.post(
        "/api/classify",
        headers=headers,
        json={"text": "test", "enabled_models": ["gemini"]},
    )
    # Assuming a successful response for a valid token, even if the classification service itself is mocked or returns an empty result
    # The actual classification logic is tested in test_classification_service.py
    assert (
        response.status_code == 200 or response.status_code == 422
    )  # 422 if validation error for request body, but 200 if it proceeds to service
    # Further assertions can be added here based on the expected response structure after successful authentication
