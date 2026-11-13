"""HTTP API and web page.

    uv run uvicorn groundwork.api:app --port 8000

Then open http://localhost:8000. Interactive API docs are at /docs.

The models are loaded once, when the server starts (the "lifespan"), and kept in memory.
That's the main difference from the command-line scripts, which pay the ~15 s loading
cost on every run.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from groundwork.config import PROJECT_ROOT
from groundwork.documents import Chunk
from groundwork.llm import LLMError
from groundwork.rag import RagPipeline, get_pipeline

WEB_DIR = PROJECT_ROOT / "web"


# --- Response shapes (also documented automatically at /docs) ---------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class ChunkOut(BaseModel):
    source_id: str
    title: str
    path: str
    location: str
    page: int | None
    section: str | None
    text: str


class RetrievedOut(BaseModel):
    rank: int
    score: float  # cosine similarity (meaning)
    keyword: float | None  # BM25 match strength 0-1, None when hybrid search is off
    used: bool  # passed the gate and was sent to the model
    chunk: ChunkOut


class SourceOut(BaseModel):
    number: int  # the [n] in the answer
    score: float
    chunk: ChunkOut


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    sources: list[SourceOut]
    retrieved: list[RetrievedOut]
    min_similarity: float
    min_keyword: float
    model: str | None
    timings: dict[str, float]


def _chunk_out(c: Chunk) -> ChunkOut:
    return ChunkOut(
        source_id=c.source_id, title=c.title, path=c.path, location=c.location,
        page=c.page, section=c.section, text=c.text,
    )


# --- App ---------------------------------------------------------------------


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    """Build the app. Tests pass a pipeline with fake models; the server builds the real one."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or get_pipeline()
        app.state.pipeline.warmup()  # load the embedding model before the first request
        yield

    app = FastAPI(title="groundwork-rag", lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict:
        p: RagPipeline = app.state.pipeline
        check = getattr(p.llm, "check", None)
        problem = check() if check else None
        return {
            "model": p.llm.name,
            "llm_ok": problem is None,
            "llm_problem": problem,
            "chunks": p.retriever.store.count(),
            "top_k": p.top_k,
            "hybrid": p.retriever.hybrid,
            "min_similarity": p.min_similarity,
            "min_keyword": p.min_keyword,
        }

    @app.get("/api/documents")
    def documents() -> list[dict]:
        return app.state.pipeline.retriever.store.documents()

    # A plain `def` (not async): FastAPI runs it in a worker thread, so a slow model
    # call doesn't freeze the server for other requests.
    @app.post("/api/ask")
    def ask(req: AskRequest) -> AskResponse:
        try:
            r = app.state.pipeline.ask(req.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AskResponse(
            question=r.question,
            answer=r.answer,
            refused=r.refused,
            refusal_reason=r.refusal_reason,
            sources=[SourceOut(number=s.number, score=s.hit.score, chunk=_chunk_out(s.hit.chunk)) for s in r.sources],
            retrieved=[
                RetrievedOut(
                    rank=i, score=h.score, keyword=h.keyword, used=h in r.context, chunk=_chunk_out(h.chunk)
                )
                for i, h in enumerate(r.retrieved, start=1)
            ],
            min_similarity=r.min_similarity,
            min_keyword=r.min_keyword,
            model=r.model,
            timings=r.timings,
        )

    # The web page. Mounted last so it doesn't shadow the /api routes.
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
