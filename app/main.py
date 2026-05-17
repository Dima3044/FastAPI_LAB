# app/main.py — начало файла
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pickle
import numpy as np
from sqlalchemy.orm import Session
from . import models, database

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Создаём таблицы ТОЛЬКО при старте приложения, не при импорте
    models.Base.metadata.create_all(bind=database.engine)
    yield
    # Опционально: очистка при завершении

app = FastAPI(title="Insurance Prediction API", lifespan=lifespan)

# Загрузка модели (остается как было)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest, db: Session = Depends(database.get_db)):
    data = np.array(request.features).reshape(1, -1)
    prediction = float(model.predict(data)[0])
    
    db_record = models.PredictionRecord(
        input_data=str(request.features),
        prediction_result=round(prediction, 2)
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {
        "prediction": round(prediction, 2), 
        "record_id": db_record.id
    }
