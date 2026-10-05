from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from app.admin.session import require_admin_session
from app.agent.engine import assistant_engine_name
from app.llm.model_status import read_model_status

router = APIRouter()


class ModelStatusResponse(BaseModel):
    chat_model: str = Field(serialization_alias="chatModel")
    embed_model: str = Field(serialization_alias="embedModel")
    chat_ready: bool = Field(serialization_alias="chatReady")
    embed_ready: bool = Field(serialization_alias="embedReady")
    ollama_reachable: bool = Field(serialization_alias="ollamaReachable")
    assistant_engine: str = Field(serialization_alias="assistantEngine")
    detail: str


@router.get("/admin/status", response_model=ModelStatusResponse)
def read_admin_status(x_admin_session: str = Header(default="")) -> ModelStatusResponse:
    require_admin_session(x_admin_session)
    status = read_model_status()
    return ModelStatusResponse(
        chat_model=status.chat_model,
        embed_model=status.embed_model,
        chat_ready=status.chat_ready,
        embed_ready=status.embed_ready,
        ollama_reachable=status.ollama_reachable,
        assistant_engine=assistant_engine_name(),
        detail=status.detail,
    )
