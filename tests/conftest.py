# tests/conftest.py
import os
import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 🚨 ВАЖНО: Установите DATABASE_URL ДО импорта app.database!
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

# Теперь можно импортировать после установки env var
from app.database import Base, get_db

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Создаём тестовый движок (отдельный от production)
test_engine = create_engine(
    "sqlite:///./test.db",
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(autouse=True)
def setup_teardown_db():
    """Создаёт таблицы перед тестом и очищает после"""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)

def override_get_db():
    """Заменяет get_db на тестовую версию"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Переопределяем зависимость в приложении
from app.main import app
app.dependency_overrides[get_db] = override_get_db
