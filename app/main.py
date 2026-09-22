"""
FastAPI skeleton - intentionally thin. Real endpoint wiring happens once
the core pipeline (app/pipeline.py) is validated against the full
hand-written + real STT-failure test sets. Do not add business logic here;
it belongs in pipeline.py so it stays independently testable/profileable.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from app.pipeline import correct_text, DEFAULT_THRESHOLD

app = FastAPI(title="Mozhi", description="Non-generative phonetic correction layer for domain-specific STT")


class CorrectionRequest(BaseModel):
    text: str
    threshold: float = DEFAULT_THRESHOLD


class MatchOut(BaseModel):
    original: str
    replacement: str
    method: str
    score: float


class CorrectionResponse(BaseModel):
    original_text: str
    corrected_text: str
    matches: list[MatchOut]
    elapsed_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/correct", response_model=CorrectionResponse)
def correct(req: CorrectionRequest) -> CorrectionResponse:
    result = correct_text(req.text, threshold=req.threshold)
    return CorrectionResponse(
        original_text=result.original_text,
        corrected_text=result.corrected_text,
        matches=[
            MatchOut(original=m.original, replacement=m.replacement, method=m.method, score=m.score)
            for m in result.matches
        ],
        elapsed_ms=result.elapsed_ms,
    )
