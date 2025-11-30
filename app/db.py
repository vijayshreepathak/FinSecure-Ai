import os
import json
from datetime import datetime
from typing import Generator
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FraudPrediction(Base):
    __tablename__ = "fraud_predictions"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    true_label = Column(Integer, nullable=True)
    amount = Column(Float, nullable=True)
    transaction_hour = Column(Integer, nullable=True)
    features_json = Column(Text, nullable=False)
    explanation_json = Column(Text, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


