from fastapi import FastAPI, HTTPException

from app.model import GreenMilePredictor
from app.schemas import (
    SoHPredictionRequest,
    SoHPredictionResponse,
)


app = FastAPI(
    title="GreenMile SoH AI Service",
    version="1.0.0",
    description=(
        "FastAPI inference service for the final GreenMile LSTM "
        "State-of-Health model."
    ),
)

predictor = GreenMilePredictor()


@app.get("/")
def root():
    return {
        "service": "GreenMile SoH AI Service",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": predictor.config["model_name"],
        "sequenceLength": predictor.sequence_length,
        "inputFeatures": predictor.feature_order,
    }


@app.get("/model-info")
def model_info():
    return {
        "model": predictor.config["model_name"],
        "inputSize": predictor.config["input_size"],
        "hiddenSize": predictor.config["hidden_size"],
        "numLayers": predictor.config["num_layers"],
        "dropout": predictor.config["dropout"],
        "sequenceLength": predictor.config["sequence_length"],
        "featureOrder": predictor.feature_order,
        "trainingBatteries": predictor.config["training_batteries"],
        "unseenTestBattery": predictor.config["unseen_test_battery"],
        "testMetrics": predictor.config["test_metrics"],
    }


@app.post(
    "/predict/soh",
    response_model=SoHPredictionResponse,
)
def predict_soh(request: SoHPredictionRequest):
    try:
        records = [
            record.model_dump()
            for record in request.records
        ]

        soh = predictor.predict(records)

        return SoHPredictionResponse(
            batteryId=request.batteryId,
            predictedSoH=round(soh, 6),
            predictedSoHPercent=round(soh * 100.0, 2),
            sequenceLength=predictor.sequence_length,
            model=predictor.config["model_name"],
            note=(
                "AI-estimated SoH. The NASA-trained model should "
                "not be treated as ground-truth SoH for a different "
                "real-world battery without validation."
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SoH prediction failed: {exc}",
        ) from exc
