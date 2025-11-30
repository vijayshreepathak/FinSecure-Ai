import os
import random
import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def autoencoder_errors(model: Autoencoder, data_array: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
        recon = model(x_tensor)
        errors = torch.mean((recon - x_tensor) ** 2, dim=1).cpu().numpy()
    return errors


def main() -> None:
    data_path = Path("creditcard.csv")
    if not data_path.exists():
        raise RuntimeError("creditcard.csv not found in project root")
    df = pd.read_csv(data_path)
    df["Amount_log"] = np.log1p(df["Amount"])
    df["transaction_hour"] = ((df["Time"] % (60 * 60 * 24)) // 3600).astype(int)
    feature_cols: List[str] = ["Time", "Amount", "Amount_log", "transaction_hour"] + [f"V{i}" for i in range(1, 29)]
    target_col = "Class"
    X = df[feature_cols]
    y = df[target_col]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    val_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=SEED
    )
    numeric_to_scale = ["Time", "Amount", "Amount_log", "transaction_hour"]
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_to_scale)],
        remainder="passthrough",
    )
    X_train_smote, y_train_smote = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    supervised_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", xgb_clf)])
    supervised_pipeline.fit(
        X_train_smote,
        y_train_smote,
        model__eval_set=[(X_val, y_val)],
        model__early_stopping_rounds=50,
        model__verbose=False,
    )
    y_val_pred_proba = supervised_pipeline.predict_proba(X_val)[:, 1]
    y_test_pred_proba = supervised_pipeline.predict_proba(X_test)[:, 1]
    print("Validation classification report")
    print(classification_report(y_val, (y_val_pred_proba >= 0.5).astype(int), digits=4))
    print("Val ROC-AUC", roc_auc_score(y_val, y_val_pred_proba))
    print("Val PR-AUC", average_precision_score(y_val, y_val_pred_proba))
    print("Test ROC-AUC", roc_auc_score(y_test, y_test_pred_proba))
    print("Test PR-AUC", average_precision_score(y_test, y_test_pred_proba))
    preprocessor_fitted = supervised_pipeline.named_steps["preprocess"]
    X_train_nonfraud = X_train[y_train == 0]
    X_train_nonfraud_trans = preprocessor_fitted.transform(X_train_nonfraud)
    X_val_trans = preprocessor_fitted.transform(X_val)
    X_test_trans = preprocessor_fitted.transform(X_test)
    iso_forest = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.001,
        random_state=SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_nonfraud_trans)
    input_dim = X_train_nonfraud_trans.shape[1]
    autoencoder = Autoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_dataset = TensorDataset(torch.tensor(X_train_nonfraud_trans, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    epochs = 20
    for epoch in range(epochs):
        autoencoder.train()
        for batch in train_loader:
            batch_x = batch[0].to(device)
            optimizer.zero_grad()
            recon = autoencoder(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
    iso_val_raw = -iso_forest.score_samples(X_val_trans)
    iso_test_raw = -iso_forest.score_samples(X_test_trans)
    ae_val_errors = autoencoder_errors(autoencoder, X_val_trans)
    ae_test_errors = autoencoder_errors(autoencoder, X_test_trans)
    iso_min = float(iso_val_raw.min())
    iso_max = float(iso_val_raw.max())
    ae_max = float(ae_val_errors.max())
    iso_val_norm = (iso_val_raw - iso_min) / (iso_max - iso_min + 1e-8)
    iso_test_norm = (iso_test_raw - iso_min) / (iso_max - iso_min + 1e-8)
    ae_val_norm = ae_val_errors / (ae_max + 1e-8)
    ae_test_norm = ae_test_errors / (ae_max + 1e-8)
    w_supervised = 0.7
    w_iso = 0.15
    w_autoencoder = 0.15
    risk_val = w_supervised * y_val_pred_proba + w_iso * iso_val_norm + w_autoencoder * ae_val_norm
    risk_test = w_supervised * y_test_pred_proba + w_iso * iso_test_norm + w_autoencoder * ae_test_norm
    nonfraud_mask = y_val == 0
    threshold = float(np.quantile(risk_val[nonfraud_mask], 0.99))
    print("Ensemble validation ROC-AUC", roc_auc_score(y_val, risk_val))
    print("Ensemble validation PR-AUC", average_precision_score(y_val, risk_val))
    print("Ensemble test ROC-AUC", roc_auc_score(y_test, risk_test))
    print("Ensemble test PR-AUC", average_precision_score(y_test, risk_test))
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(supervised_pipeline, models_dir / "supervised_pipeline.joblib")
    joblib.dump(preprocessor_fitted, models_dir / "preprocessor.joblib")
    joblib.dump(iso_forest, models_dir / "iso_forest.joblib")
    torch.save(autoencoder.state_dict(), models_dir / "autoencoder.pt")
    config = {
        "threshold": threshold,
        "w_supervised": w_supervised,
        "w_iso": w_iso,
        "w_autoencoder": w_autoencoder,
        "iso_min": float(iso_min),
        "iso_max": float(iso_max),
        "ae_max": float(ae_max),
        "feature_cols": feature_cols,
        "numeric_to_scale": numeric_to_scale,
    }
    with open(models_dir / "config.json", "w") as f:
        json.dump(config, f)
    background = X_train.sample(n=min(200, len(X_train)), random_state=SEED)
    joblib.dump(background, models_dir / "shap_background.joblib")
    print("Artifacts saved to models directory")


if __name__ == "__main__":
    main()


