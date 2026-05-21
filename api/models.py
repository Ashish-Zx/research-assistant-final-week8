# api/models.py
from pydantic import BaseModel


class AgentRequest(BaseModel):
    query: str


class RatingRequest(BaseModel):
    trace_id: str
    rating: int
