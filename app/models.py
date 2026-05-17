from sqlalchemy import Column, Integer, Float, String
from app.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(String, nullable=False)
    prediction_result = Column(Float, nullable=False)