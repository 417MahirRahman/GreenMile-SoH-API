from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"

CHECKPOINT_PATH = MODEL_DIR / "greenmile_lstm.pth"
SCALER_PATH = MODEL_DIR / "greenmile_scaler.pkl"
CONFIG_PATH = MODEL_DIR / "greenmile_model_config.json"
FEATURE_ORDER_PATH = MODEL_DIR / "greenmile_feature_order.json"


class GreenMileLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden).squeeze(-1)


class GreenMilePredictor:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config: dict[str, Any] = json.load(f)

        with open(FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
            self.feature_order: list[str] = json.load(f)

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=self.device,
            )

        self.sequence_length = int(
            checkpoint.get(
                "sequence_length",
                self.config["sequence_length"],
            )
        )

        self.model = GreenMileLSTM(
            input_size=int(
                checkpoint.get(
                    "input_size",
                    self.config["input_size"],
                )
            ),
            hidden_size=int(
                checkpoint.get(
                    "hidden_size",
                    self.config["hidden_size"],
                )
            ),
            num_layers=int(
                checkpoint.get(
                    "num_layers",
                    self.config["num_layers"],
                )
            ),
            dropout=float(
                checkpoint.get(
                    "dropout",
                    self.config["dropout"],
                )
            ),
        )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        if len(self.feature_order) != int(self.config["input_size"]):
            raise RuntimeError(
                "Feature-order length does not match model input size."
            )

        scaler_features = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )
        if scaler_features is not None and int(scaler_features) != len(
            self.feature_order
        ):
            raise RuntimeError(
                "Scaler feature count does not match model feature count."
            )

    def predict(self, records: list[dict[str, float]]) -> float:
        if len(records) != self.sequence_length:
            raise ValueError(
                f"Expected exactly {self.sequence_length} cycle records, "
                f"but received {len(records)}."
            )

        # DataFrame preserves the exact training feature order.
        frame = pd.DataFrame(
            records,
            columns=self.feature_order,
        )

        if frame.isnull().any().any():
            raise ValueError("Input contains a missing feature value.")

        values = frame.to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError("All feature values must be finite numbers.")

        scaled = self.scaler.transform(frame)
        sequence = np.asarray(
            scaled,
            dtype=np.float32,
        )[None, :, :]

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            prediction = self.model(tensor).item()

        return float(prediction)
