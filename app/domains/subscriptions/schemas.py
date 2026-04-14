from datetime import datetime

from pydantic import BaseModel


class PlanResponse(BaseModel):
    id: str
    name: str
    display_name: str
    chat_daily_limit: int | None
    inference_daily_limit: int | None
    api_monthly_limit: int | None


class SubscribeRequest(BaseModel):
    plan_name: str


class SubscriptionResponse(BaseModel):
    id: str
    plan: PlanResponse
    started_at: datetime
    expires_at: datetime | None
    is_active: bool
