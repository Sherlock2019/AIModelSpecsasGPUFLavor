"""FastAPI application entry point: `uvicorn app.main:app --reload`."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import router
from .db import SessionLocal, init_db
from .repository import seed


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as session:
        seed(session)
    yield


app = FastAPI(
    title="AI Model → GPU Flavor Calculator",
    version="2.0.0",
    description=(
        "Size LLMs, AI VM flavors and custom ML models to GPU memory, NVIDIA/AMD GPUs (vGPU, dedicated, multi-GPU), "
        "AI flavors and OpenStack flavor recommendations."
    ),
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


class SPAStaticFiles(StaticFiles):
    """Serves the built frontend; unknown non-API paths fall back to index.html (client-side routes)."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not path.startswith("api/"):
                return await super().get_response("index.html", scope)
            raise


# Single-process mode (start.sh): serve the web UI from the same port as the API.
# Mounted last so /api, /docs, /openapi.json and /healthz keep priority.
_static_dir = Path(os.getenv("GPUCALC_STATIC_DIR", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if (_static_dir / "index.html").is_file():
    app.mount("/", SPAStaticFiles(directory=_static_dir, html=True), name="web")
