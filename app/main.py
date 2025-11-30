import io
import json
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from .db import get_db, init_db, FraudPrediction
from .model import FraudModelService
from .schemas import TransactionIn, PredictionOut, BatchPredictionOut, MetricsOut, AlertOut

app = FastAPI(title="FinSecure AI Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service: Optional[FraudModelService] = None


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    global model_service
    if model_service is None:
        model_service = FraudModelService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(transaction: TransactionIn, db: Session = Depends(get_db)) -> PredictionOut:
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    try:
        payload = transaction.dict()
        result = model_service.predict_single(payload)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
    features_to_store = dict(payload)
    features_to_store.pop("Class", None)
    record = FraudPrediction(
        probability=result["probability"],
        risk_score=result["risk_score"],
        threshold=result["threshold"],
        is_fraud=bool(result["label"]),
        true_label=payload.get("Class"),
        amount=features_to_store.get("Amount"),
        transaction_hour=int((features_to_store.get("Time", 0.0) % (60 * 60 * 24)) // 3600),
        features_json=json.dumps(features_to_store),
        explanation_json=json.dumps(result["explanations"]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return PredictionOut(**result)


@app.post("/predict_batch", response_model=BatchPredictionOut)
async def predict_batch(
    file: UploadFile = File(default=None),
    items: Optional[List[TransactionIn]] = Body(default=None),
    db: Session = Depends(get_db),
) -> BatchPredictionOut:
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    records: List[dict] = []
    if file is not None:
        content = await file.read()
        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid CSV file")
        if "Time" not in df.columns or "Amount" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV missing required columns")
        df = df.fillna(0.0)
        for _, row in df.iterrows():
            rec = {}
            for col in df.columns:
                if col == "Class":
                    if not pd.isna(row[col]):
                        rec["Class"] = int(row[col])
                else:
                    rec[col] = float(row[col])
            records.append(rec)
    elif items is not None:
        for item in items:
            records.append(item.dict())
    else:
        raise HTTPException(status_code=400, detail="No data provided")
    outputs = []
    for rec in records:
        result = model_service.predict_single(rec)
        outputs.append(result)
        features_to_store = dict(rec)
        features_to_store.pop("Class", None)
        record = FraudPrediction(
            probability=result["probability"],
            risk_score=result["risk_score"],
            threshold=result["threshold"],
            is_fraud=bool(result["label"]),
            true_label=rec.get("Class"),
            amount=features_to_store.get("Amount"),
            transaction_hour=int((features_to_store.get("Time", 0.0) % (60 * 60 * 24)) // 3600),
            features_json=json.dumps(features_to_store),
            explanation_json=json.dumps(result["explanations"]),
        )
        db.add(record)
    db.commit()
    return BatchPredictionOut(results=[PredictionOut(**r) for r in outputs], count=len(outputs))


@app.get("/alerts", response_model=List[AlertOut])
def alerts(limit: int = 50, db: Session = Depends(get_db)) -> List[AlertOut]:
    rows = (
        db.query(FraudPrediction)
        .filter(FraudPrediction.is_fraud.is_(True))
        .order_by(FraudPrediction.created_at.desc())
        .limit(limit)
        .all()
    )
    out: List[AlertOut] = []
    for r in rows:
        out.append(
            AlertOut(
                id=r.id,
                created_at=r.created_at.isoformat(),
                amount=float(r.amount) if r.amount is not None else 0.0,
                risk_score=float(r.risk_score),
                probability=float(r.probability),
                threshold=float(r.threshold),
            )
        )
    return out


@app.get("/metrics", response_model=MetricsOut)
def metrics(db: Session = Depends(get_db)) -> MetricsOut:
    total = db.query(func.count(FraudPrediction.id)).scalar() or 0
    fraud_count = db.query(func.count(FraudPrediction.id)).filter(FraudPrediction.is_fraud.is_(True)).scalar() or 0
    fraud_rate = float(fraud_count) / float(total) if total > 0 else 0.0
    rows = db.query(FraudPrediction).all()
    tp = 0
    fp = 0
    tn = 0
    fn_ = 0
    for r in rows:
        if r.true_label is None:
            continue
        pred = 1 if r.is_fraud else 0
        true = int(r.true_label)
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 1 and pred == 0:
            fn_ += 1
    precision = None
    recall = None
    f1 = None
    if tp + fp > 0:
        precision = float(tp) / float(tp + fp)
    if tp + fn_ > 0:
        recall = float(tp) / float(tp + fn_)
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    by_hour_map = {}
    amount_bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
    amount_hist_map = {}
    for r in rows:
        hour = r.transaction_hour if r.transaction_hour is not None else 0
        key = int(hour)
        if key not in by_hour_map:
            by_hour_map[key] = {"hour": key, "fraud": 0, "non_fraud": 0}
        if r.is_fraud:
            by_hour_map[key]["fraud"] += 1
        else:
            by_hour_map[key]["non_fraud"] += 1
        amt = float(r.amount) if r.amount is not None else 0.0
        bin_label = None
        for i in range(len(amount_bins) - 1):
            if amount_bins[i] <= amt < amount_bins[i + 1]:
                bin_label = f"{amount_bins[i]}-{amount_bins[i+1]}"
                break
        if bin_label is None:
            bin_label = f">={amount_bins[-1]}"
        if bin_label not in amount_hist_map:
            amount_hist_map[bin_label] = {"bin": bin_label, "fraud": 0, "non_fraud": 0}
        if r.is_fraud:
            amount_hist_map[bin_label]["fraud"] += 1
        else:
            amount_hist_map[bin_label]["non_fraud"] += 1
    by_hour = list(sorted(by_hour_map.values(), key=lambda x: x["hour"]))
    amount_hist = list(amount_hist_map.values())
    threshold = float(model_service.config["threshold"]) if model_service is not None else 0.5
    return MetricsOut(
        total=int(total),
        fraud_count=int(fraud_count),
        fraud_rate=float(fraud_rate),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn_),
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        by_hour=by_hour,
        amount_hist=amount_hist,
    )


