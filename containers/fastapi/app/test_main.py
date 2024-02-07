from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome from the API"}


def test_predict_spam():
    response = client.post(
        "/predict/",
        json={"sms": "Hello world, I am a spam!"},
    )
    assert response.status_code == 200
    assert response.json() == {"label": "spam"}


def test_predict_non_spam():
    response = client.post(
        "/predict/",
        json={"sms": "Hello world, I am a NOT a spam!"},
    )
    assert response.status_code == 200
    assert response.json() == {"label": "non spam"}
