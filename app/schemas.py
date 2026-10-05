from pydantic import BaseModel, Field, field_validator


class CycleFeatureRecord(BaseModel):
    cycle_input_raw: float = Field(
        ...,
        description="Raw chronological operation/cycle index."
    )
    voltage_mean_v: float = Field(
        ...,
        description="Mean discharge-cycle voltage in volts."
    )
    voltage_min_v: float = Field(
        ...,
        description="Minimum discharge-cycle voltage in volts."
    )
    current_abs_mean_a: float = Field(
        ...,
        description="Mean absolute discharge current in amperes."
    )
    temperature_mean_c: float = Field(
        ...,
        description="Mean discharge-cycle battery temperature in °C."
    )
    temperature_max_c: float = Field(
        ...,
        description="Maximum discharge-cycle battery temperature in °C."
    )
    discharge_duration_s: float = Field(
        ...,
        description="Discharge-cycle duration in seconds."
    )


class SoHPredictionRequest(BaseModel):
    batteryId: str | None = Field(
        default=None,
        description="Optional GreenMile battery identifier."
    )
    records: list[CycleFeatureRecord] = Field(
        ...,
        description="Exactly 10 chronological cycle-level records."
    )

    @field_validator("records")
    @classmethod
    def validate_records(cls, value):
        if len(value) != 10:
            raise ValueError(
                "GreenMile LSTM requires exactly 10 chronological "
                "cycle-level records."
            )
        return value


class SoHPredictionResponse(BaseModel):
    batteryId: str | None
    predictedSoH: float
    predictedSoHPercent: float
    sequenceLength: int
    model: str
    note: str
