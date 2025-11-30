import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from fastapi import HTTPException
import shap


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class FraudModelService:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config()
        self.feature_cols = self.config["feature_cols"]
        self.numeric_to_scale = self.config["numeric_to_scale"]
        self.supervised_pipeline = self._load_joblib("supervised_pipeline.joblib")
        self.preprocessor = self._load_joblib("preprocessor.joblib")
        self.iso_forest = self._load_joblib("iso_forest.joblib")
        self.background = self._load_joblib("shap_background.joblib")
        self.autoencoder = self._load_autoencoder()
        self._init_explainer()

    def _load_config(self) -> Dict[str, Any]:
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise HTTPException(status_code=500, detail="Model configuration not found")
        with open(config_path, "r") as f:
            return json.load(f)

    def _load_joblib(self, name: str):
        path = self.model_dir / name
        if not path.exists():
            raise HTTPException(status_code=500, detail=f"Artifact {name} not found")
        return joblib.load(path)

    def _load_autoencoder(self) -> Autoencoder:
        path = self.model_dir / "autoencoder.pt"
        if not path.exists():
            raise HTTPException(status_code=500, detail="Autoencoder weights not found")
        if not isinstance(self.background, pd.DataFrame):
            raise HTTPException(status_code=500, detail="SHAP background data is invalid")
        sample_df = self.background[self.feature_cols].iloc[[0]]
        sample = self.preprocessor.transform(sample_df)
        input_dim = int(sample.shape[1])
        model = Autoencoder(input_dim=input_dim).to(self.device)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _init_explainer(self) -> None:
        self.explainer = None
        try:
            self.explainer = shap.Explainer(self.supervised_pipeline, self.background)
        except Exception:
            try:
                xgb_model = getattr(self.supervised_pipeline, "named_steps", {}).get("model")
                if xgb_model is None:
                    self.explainer = None
                    return
                _ = self.preprocessor.transform(self.background[self.feature_cols])
                self.explainer = shap.TreeExplainer(xgb_model)
            except Exception:
                self.explainer = None

    def _ensure_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        d = dict(data)
        if "Amount_log" not in d:
            d["Amount_log"] = float(np.log1p(d["Amount"]))
        if "transaction_hour" not in d:
            d["transaction_hour"] = int((d["Time"] % (60 * 60 * 24)) // 3600)
        df = pd.DataFrame([d])
        df = df[self.feature_cols]
        return df

    def _autoencoder_error(self, x_array: np.ndarray) -> float:
        with torch.no_grad():
            x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)
            recon = self.autoencoder(x_tensor)
            errors = torch.mean((recon - x_tensor) ** 2, dim=1).cpu().numpy()
        return float(errors[0])

    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = self._ensure_features(data)
        prob = float(self.supervised_pipeline.predict_proba(df)[:, 1][0])
        x_trans = self.preprocessor.transform(df)
        iso_raw = float(-self.iso_forest.score_samples(x_trans)[0])
        iso_min = float(self.config["iso_min"])
        iso_max = float(self.config["iso_max"])
        iso_norm = (iso_raw - iso_min) / (iso_max - iso_min + 1e-8)
        ae_error = self._autoencoder_error(x_trans)
        ae_max = float(self.config["ae_max"])
        ae_norm = ae_error / (ae_max + 1e-8)
        w_supervised = float(self.config["w_supervised"])
        w_iso = float(self.config["w_iso"])
        w_autoencoder = float(self.config["w_autoencoder"])
        risk_score = w_supervised * prob + w_iso * iso_norm + w_autoencoder * ae_norm
        threshold = float(self.config["threshold"])
        label = int(risk_score >= threshold)
        try:
            explanations: List[Dict[str, Any]] = []
            if self.explainer is not None:
                try:
                    shap_values = self.explainer(df)
                    vals = shap_values.values[0]
                except Exception:
                    x_trans_for_shap = self.preprocessor.transform(df)
                    shap_values_arr = self.explainer.shap_values(x_trans_for_shap)  # type: ignore[attr-defined]
                    vals = shap_values_arr[0]
            abs_vals = np.abs(vals)
            top_idx = np.argsort(abs_vals)[-3:][::-1]
            for idx in top_idx:
                explanations.append(
                    {
                        "feature": self.feature_cols[idx],
                        "value": float(df.iloc[0, idx]),
                        "contribution": float(vals[idx]),
                    }
                )
        except Exception:
            explanations = []
        return {
            "probability": prob,
            "risk_score": float(risk_score),
            "label": label,
            "threshold": threshold,
            "iso_score_raw": iso_raw,
            "iso_norm": float(iso_norm),
            "ae_error": float(ae_error),
            "ae_norm": float(ae_norm),
            "explanations": explanations,
        }

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for rec in records:
            outputs.append(self.predict_single(rec))
        return outputs


