from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.sdr_agent import SDRAgent
from app.api.routes import build_router
from app.config import settings
from app.db.store import LeadStore
from app.llm.factory import build_llm
from app.tools.crm_tool import CRMTool
from app.tools.research_tool import ResearchTool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifecycle hook.

    We eagerly initialize DB schema at startup so first request latency is lower
    and schema issues fail early.
    """
    await app.state.store.initialize()
    yield


def create_app() -> FastAPI:
    """Compose application dependencies and register routes/static assets."""
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    store = LeadStore(settings.db_path)
    llm = build_llm()
    crm = CRMTool(store=store)
    researcher = ResearchTool()
    agent = SDRAgent(llm=llm, crm=crm, researcher=researcher)

    app.state.store = store

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.include_router(build_router(agent, store))
    return app


app = create_app()
