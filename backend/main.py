"""
FastAPI entrypoint for VibeFinder Agent.

Mounts Strawberry GraphQL at /graphql.
Serves GraphiQL playground at /graphql (GET).
CORS configured for Next.js dev server on localhost:3000.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the ChromaDB vibe index at startup so first query is fast."""
    logger.info("VibeFinder Agent starting up...")
    try:
        from backend.tools.vibe_search import build_vibe_index
        build_vibe_index()
        logger.info("Vibe index ready.")
    except Exception as e:
        logger.warning(f"Vibe index build failed (non-fatal): {e}")
    yield
    logger.info("VibeFinder Agent shutting down.")


from backend.schema import schema
from backend.streaming import router as stream_router

graphql_router = GraphQLRouter(schema)

app = FastAPI(
    title="VibeFinder Agent API",
    description="Agentic music recommender powered by LangGraph + Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graphql_router, prefix="/graphql")
app.include_router(stream_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "vibefinder-agent"}
