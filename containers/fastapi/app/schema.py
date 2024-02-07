from pydantic import BaseModel


class Sms(BaseModel):
    sms: str


class PredictionOut(BaseModel):
    label: str
