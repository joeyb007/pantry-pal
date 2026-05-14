from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def make_mock_model_and_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = MagicMock()
    mock_tokenizer.decode.return_value = "Gluten-Free Pasta\n\nIngredients:\n- rice pasta\n\nDirections:\nBoil pasta."
    mock_tokenizer.eos_token_id = 0
    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cpu"
    return mock_model, mock_tokenizer


def test_health_returns_ok():
    with patch("main.load_model", return_value=make_mock_model_and_tokenizer()):
        from main import app
        client = TestClient(app)
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_check_constraints_returns_flags():
    mock_xgb = {
        "gluten": {"probability": 0.05, "present": False},
        "dairy": {"probability": 0.9, "present": True},
    }
    with patch("main.classify_ingredients", return_value=mock_xgb):
        with patch("main.load_model", return_value=make_mock_model_and_tokenizer()):
            from main import app
            client = TestClient(app)
            response = client.post("/check_constraints", json={"ingredients": ["milk", "flour"]})
    assert response.status_code == 200
    flags = {f["flag"]: f for f in response.json()["flags"]}
    assert flags["dairy"]["present"] is True
    assert flags["gluten"]["present"] is False


def test_generate_returns_recipe_and_constraint_check():
    mock_xgb = {"gluten": {"probability": 0.05, "present": False}}
    mock_recipe = "Gluten-Free Pasta\n\nIngredients:\n- rice pasta\n\nDirections:\nBoil pasta."
    with patch("main.classify_ingredients", return_value=mock_xgb):
        with patch("main.load_model", return_value=make_mock_model_and_tokenizer()):
            with patch("main._generate", return_value=mock_recipe):
                from main import app
                client = TestClient(app)
                response = client.post("/generate", json={
                    "restrictions": ["gluten-free"],
                    "pantry_items": ["rice pasta", "tomatoes"],
                })
    assert response.status_code == 200
    body = response.json()
    assert "recipe" in body
    assert "constraint_check" in body
