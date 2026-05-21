# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db
from app import models

client = TestClient(app)

def test_predict_valid_input():
    """Базовый тест: валидные 11 признаков"""
    payload = {
        "features": [-1.4, 0.5, -0.9, 1.0, 0.0, 0.1, 0.5, 0.2, 0.8, 0, 1]
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "record_id" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0  # возраст раковины не может быть <= 0
    assert isinstance(data["record_id"], int)


def test_predict_different_sex_values():
    """Проверка разных значений пола (sex_I, sex_M)"""
    test_cases = [
        {"features": [-1.4, 0.5, -0.9, 1.0, 0.0, 0.1, 0.5, 0.2, 0.8, 0, 0]},  # sex=F (baseline)
        {"features": [-1.4, 0.5, -0.9, 1.0, 0.0, 0.1, 0.5, 0.2, 0.8, 1, 0]},  # sex=I
        {"features": [-1.4, 0.5, -0.9, 1.0, 0.0, 0.1, 0.5, 0.2, 0.8, 0, 1]},  # sex=M
    ]
    
    for payload in test_cases:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)


def test_predict_extreme_values():
    """Тест с экстремальными, но валидными значениями признаков"""
    payload = {
        "features": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1, 1]  # большие отрицательные/положительные
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["prediction"], float)
    # Предсказание может быть любым числом, главное — не ошибка


def test_predict_zeros():
    """Тест с нулевыми значениями (граничный случай)"""
    payload = {"features": [0.0] * 11}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data



def test_predict_wrong_feature_type():
    """Ошибка: нечисловые значения в features"""
    payload = {"features": [-1.4, "invalid", -0.9, 1.0, 0.0, 0.1, 0.5, 0.2, 0.8, 0, 1]}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422


def test_predict_missing_features_field():
    """Ошибка: отсутствует поле features"""
    payload = {"wrong_field": [1, 2, 3]}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422


def test_predict_empty_body():
    """Ошибка: пустое тело запроса"""
    response = client.post("/predict", json={})
    assert response.status_code == 422

