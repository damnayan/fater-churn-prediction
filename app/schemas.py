from pydantic import BaseModel, Field


class UserFeatures(BaseModel):
    baby_age_months: float = Field(
        ...,
        alias="ETA_MM_BambinoTODAY",
        ge=0,
        le=120,
        description="Baby age in months",
        examples=[14.5],
    )
    total_points: float = Field(
        ..., ge=0, description="Total loyalty points accumulated", examples=[450.0]
    )
    total_codes: int = Field(
        ..., ge=0, description="Total scanned product codes", examples=[12]
    )
    total_logins: int = Field(
        ..., ge=0, description="Total app logins count", examples=[8]
    )
    total_missions: int = Field(
        ..., ge=0, description="Total completed missions", examples=[3]
    )
    tenure_days: float = Field(
        ..., ge=0, description="Days since user registration", examples=[180.0]
    )

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    churn_probability: float
    is_churn: int
    risk_segment: str


class BatchPredictionRequest(BaseModel):
    users: list[UserFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
