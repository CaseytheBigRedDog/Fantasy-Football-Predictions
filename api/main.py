"""
main.py - the FastAPI web service.

Run it from the api/ folder:   uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs to try every endpoint in your browser.
"""
from pathlib import Path

from dotenv import load_dotenv

# Read settings (like ANTHROPIC_API_KEY) from api/.env, if that file exists.
load_dotenv(Path(__file__).resolve().parent / ".env")

import os  # noqa: E402

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import assistant  # noqa: E402
import projections as pj  # noqa: E402

app = FastAPI(title="Fantasy Football Predictions API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


def _data_or_503():
    try:
        return pj.load()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    _data_or_503()
    info = pj.meta()
    info["assistant_mode"] = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "rules"
    return info


@app.get("/api/projections")
def projections(
    position: str | None = Query(None, pattern="^(QB|RB|WR|TE)$"),
    team: str | None = Query(None, max_length=3),
    q: str | None = Query(None, max_length=50, description="Search by player name"),
    limit: int = Query(500, ge=1, le=1000),
):
    _data_or_503()
    return pj.search(q=q, position=position, team=team, limit=limit)


@app.get("/api/players/{player_id}")
def player(player_id: str):
    _data_or_503()
    row = pj.get_by_id(player_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return row


@app.get("/api/compare")
def compare(ids: list[str] = Query(..., min_length=2, max_length=8)):
    """Compare players by id (e.g. /api/compare?ids=00-0036900&ids=00-0039139)."""
    _data_or_503()
    rows = [pj.get_by_id(i) for i in ids]
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        raise HTTPException(status_code=404, detail="Need at least two known players")
    text, ranked = assistant.compare(rows)
    return {"answer": text, "players": ranked}


@app.post("/api/assistant")
def ask(body: AskRequest):
    _data_or_503()
    return assistant.ask(body.question)
