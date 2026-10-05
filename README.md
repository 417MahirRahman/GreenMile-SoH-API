# GreenMile SoH AI Service

This folder contains the complete FastAPI inference service for the final
GreenMile LSTM battery State-of-Health model.

## 1. Create a Python environment

Windows PowerShell:

```powershell
cd GreenMile-AI-Service
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Start FastAPI

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/model-info
- http://127.0.0.1:8000/docs

## 3. Prediction endpoint

`POST /predict/soh`

The model requires **exactly 10 chronological cycle-level records**.

Each record must contain these 7 features in the training schema:

1. `cycle_input_raw`
2. `voltage_mean_v`
3. `voltage_min_v`
4. `current_abs_mean_a`
5. `temperature_mean_c`
6. `temperature_max_c`
7. `discharge_duration_s`

Example body is available in `sample_request.json`.

## 4. Important integration note

The deployed NASA-trained model does not take a single instantaneous
INA219/DS18B20 reading directly.

Your Node/Express/backend pipeline must first produce cycle-level summaries
and retain the latest 10 chronological cycle records. Those 10 records are
sent to FastAPI.

For GreenMile's prototype battery, the model output should be shown as
**AI Estimated SoH** until it is validated against measured usable capacity.

## 5. Model package

The `model/` folder contains:

- `greenmile_lstm.pth`
- `greenmile_scaler.pkl`
- `greenmile_model_config.json`
- `greenmile_feature_order.json`
- `greenmile_training_summary.json`
