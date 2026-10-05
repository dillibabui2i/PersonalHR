import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.routes.activity import router as activity_router
from app.routes.chat import router as chat_router
from app.routes.extension import router as extension_router
from app.routes.documents import router as documents_router
from app.routes.organizations import router as organizations_router
from app.routes.quality import router as quality_router
from app.routes.session import router as session_router
from app.routes.status import router as status_router
from app.routes.model_admin import router as model_admin_router
from app.routes.users import router as users_router
from app.llm.model_status import read_model_status, warm_models
from app.quality.service import stop_abandoned_runs
from app.rag.retrieval.reranker import load_reranker
from app.core.settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
request_log = logging.getLogger("personal_hr")


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    stop_abandoned_runs()
    status = read_model_status()
    request_log.info("%s: %s", app.title, status.detail)
    await asyncio.to_thread(load_reranker)
    request_log.info("%s: reranker %s loaded", app.title, load_settings().reranker_model)
    try:
        await asyncio.to_thread(warm_models)
        request_log.info("%s: Ollama models warmed", app.title)
    except OSError:
        request_log.warning("%s: model warm-up was unavailable", app.title)
    yield


app = FastAPI(title="Personal HR", lifespan=application_lifespan)
app.include_router(session_router)
app.include_router(organizations_router)
app.include_router(documents_router)
app.include_router(quality_router)
app.include_router(chat_router)
app.include_router(activity_router)
app.include_router(extension_router)
app.include_router(status_router)
app.include_router(model_admin_router)
app.include_router(users_router)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    response = await call_next(request)
    request_log.info("%s %s %s", request.method, request.url.path, response.status_code)
    return response
