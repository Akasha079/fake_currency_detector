from pydantic import BaseModel

class CurrencyResponse(BaseModel):
    label: str
    confidence: float
