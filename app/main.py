from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .alignment import align_translation
from .segmenter import Config, solve
from .speakers import SpeakerError, assign_speakers
from .terminology import TermError, review_terms

app = FastAPI(title="字幕断句", version="1.6.0")

STATIC_DIR = Path(__file__).parent / "static"


class WordIn(BaseModel):
    text: str
    start: float
    end: float


class BoundaryIn(BaseModel):
    """词间隙边界约束: index = k 表示词 k-1 与词 k 之间的间隙(1 <= k <= len(words)-1)。"""
    index: int
    type: Literal["must_break", "no_break"]


class LockIn(BaseModel):
    """锁定段: 固定词范围 [word_range[0], word_range[1]](含端点)与内部换行。

    line_break = k 表示段内换行落在词 k-1 与词 k 之间(必须是段内词间隙,
    且不得位于不可拆短语内部); 为 null 表示锁定为单行。
    """
    word_range: list[int] = Field(min_length=2, max_length=2)
    line_break: int | None = None


class SegmentRequest(BaseModel):
    words: list[WordIn]
    phrases: list[list[int]] = Field(default_factory=list)
    shots: list[float] = Field(default_factory=list)
    boundaries: list[BoundaryIn] = Field(default_factory=list)
    locks: list[LockIn] = Field(default_factory=list)
    alternatives: int = Field(default=1, ge=1, le=5)
    config: dict = Field(default_factory=dict)


class TermIn(BaseModel):
    """术语表条目: canonical 为规范写法, variants 为多个识别变体(录入原文)。"""
    canonical: str
    variants: list[str] = Field(default_factory=list)


class ReviewSegmentIn(BaseModel):
    """当前选中的断句方案中的一段: 词范围与段内换行(词间隙下标, null=单行)。"""
    word_range: list[int] = Field(min_length=2, max_length=2)
    line_break: int | None = None


class ReviewRequest(BaseModel):
    words: list[WordIn]
    segments: list[ReviewSegmentIn]
    terms: list[TermIn] = Field(default_factory=list)


class AlignSegmentIn(BaseModel):
    """当前选中的断句方案中的一段: 词范围(对齐据此还原源段文本与时间)。"""
    word_range: list[int] = Field(min_length=2, max_length=2)
    line_break: int | None = None


class AlignRequest(BaseModel):
    words: list[WordIn]
    segments: list[AlignSegmentIn] = Field(default_factory=list)
    translation: str = ""


class SpeakerSegmentIn(BaseModel):
    """当前选中的断句方案中的一段: 词范围(说话人标注据此还原段时间; line_break 不使用)。"""
    word_range: list[int] = Field(min_length=2, max_length=2)
    line_break: int | None = None


class SpeakerOverrideIn(BaseModel):
    """人工覆盖: 把第 segment 段(0 起)的标签钉死为 label(说话人名或 "unknown")。"""
    segment: int = Field(ge=0)
    label: str


class SpeakerRequest(BaseModel):
    words: list[WordIn]
    segments: list[SpeakerSegmentIn] = Field(default_factory=list)
    speakers: str = ""
    change_penalty: float = Field(default=1.0, ge=0)
    coverage_threshold: float = Field(default=0.5, ge=0, le=1)
    overrides: list[SpeakerOverrideIn] = Field(default_factory=list)


@app.post("/api/segment")
def segment(req: SegmentRequest):
    cfg_fields = {k: v for k, v in req.config.items() if k in Config.__dataclass_fields__}
    cfg = Config(**cfg_fields)
    try:
        return solve([w.model_dump() for w in req.words], req.phrases, req.shots, cfg,
                     boundaries=[b.model_dump() for b in req.boundaries],
                     alternatives=req.alternatives,
                     locks=[lk.model_dump() for lk in req.locks])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/terms/review")
def terms_review(req: ReviewRequest):
    try:
        return review_terms(
            [w.model_dump() for w in req.words],
            [s.model_dump() for s in req.segments],
            [t.model_dump() for t in req.terms],
        )
    except TermError as e:
        raise HTTPException(status_code=422, detail=e.payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/align")
def align(req: AlignRequest):
    try:
        return align_translation(
            [w.model_dump() for w in req.words],
            [s.model_dump() for s in req.segments],
            req.translation,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/speakers/assign")
def speakers_assign(req: SpeakerRequest):
    try:
        return assign_speakers(
            [w.model_dump() for w in req.words],
            [s.model_dump() for s in req.segments],
            req.speakers,
            change_penalty=req.change_penalty,
            coverage_threshold=req.coverage_threshold,
            overrides=[o.model_dump() for o in req.overrides],
        )
    except SpeakerError as e:
        raise HTTPException(status_code=422, detail=e.payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
