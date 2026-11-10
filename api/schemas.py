from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

WHORegion = Literal["Africa", "Americas", "Eastern Mediterranean", "Europe", "South-East Asia", "Western Pacific"]


class Country(BaseModel):
    """One country's latest COVID-19 numbers (the model's input features)."""
    model_config = ConfigDict(populate_by_name=True)
    country: Optional[str] = None
    confirmed: int = Field(ge=0)
    deaths: int = Field(ge=0)
    new_cases: int = Field(ge=0)
    new_deaths: int = Field(ge=0)
    new_recovered: int = Field(ge=0)
    deaths_per_100_cases: float = Field(ge=0, le=100)
    confirmed_last_week: int = Field(ge=0)
    who_region: WHORegion


class Prediction(BaseModel):
    country: Optional[str] = None
    predicted_recovered: int
    predicted_recovery_pct: Optional[float] = None  # predicted_recovered / confirmed * 100


class BatchRequest(BaseModel):
    countries: List[Country]


class BatchResponse(BaseModel):
    predictions: List[Prediction]
