"""FastAPI server: upload a file, get a warehouse; then chat with it.

    POST   /api/datasets              multipart upload -> ingest + profile
    GET    /api/datasets              list what has been uploaded
    GET    /api/datasets/{id}         derived catalog + opening questions
    GET    /api/datasets/{id}/preview sample rows for the schema explorer
    GET    /api/datasets/{id}/export  guarded CSV export of a result
    DELETE /api/datasets/{id}         remove the warehouse and the originals
    POST   /api/chat                  one turn, streamed back as NDJSON

Nothing is bundled: with no upload there are no tables, no catalog, and the chat
endpoint has nothing to point at.

The chat endpoint is a POST because a turn carries the conversation with it —
the client sends the last few exchanges and the server keeps no session, so a
restart loses nothing and two tabs cannot tread on each other. The pipeline is
sync (blocking LLM calls), so it runs in a worker thread and pushes events onto
an asyncio queue the response drains as newline-delimited JSON.

    uvicorn server:app --reload
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import profiler
from agents import SuggestAgent, suggest_offline
from context import Budget
from llm import QwenClient
from models import Exchange, Metrics
from orchestrator import ChatPipeline
from warehouse import SUPPORTED, IngestError, QueryError, Warehouse, guard, ingest

load_dotenv()

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_DIR", HERE / "data"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "512")) * 1024 * 1024
PREVIEW_ROWS = 50
CHUNK = 1024 * 1024

app = FastAPI(title="DataScribe — chat with an uploaded dataset")
HISTORY_TURNS = 6       # how much conversation a turn carries with it


class Turn(BaseModel):
    question: str = ""
    answer: str = ""
    sql: str = ""
    columns: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    dataset: str
    message: str
    history: list[Turn] = Field(default_factory=list)
    max_context: int = 16000
    max_repairs: int = 3


# --- dataset storage --------------------------------------------------------
def _dir(dataset_id: str) -> Path:
    """Resolve a dataset directory, refusing anything that escapes DATA_ROOT."""
    if not dataset_id or not all(c.isalnum() or c == "-" for c in dataset_id):
        raise HTTPException(400, "bad dataset id")
    path = (DATA_ROOT / dataset_id).resolve()
    if not str(path).startswith(str(DATA_ROOT.resolve())):
        raise HTTPException(400, "bad dataset id")
    return path


def _meta_path(dataset_id: str) -> Path:
    return _dir(dataset_id) / "meta.json"


def _read_meta(dataset_id: str) -> dict:
    path = _meta_path(dataset_id)
    if not path.exists():
        raise HTTPException(404, "dataset not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_meta(dataset_id: str, meta: dict) -> None:
    _meta_path(dataset_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _open(dataset_id: str) -> tuple[Warehouse, "profiler.Catalog", dict]:
    meta = _read_meta(dataset_id)
    base = _dir(dataset_id)
    wh = Warehouse(base / "warehouse.duckdb")
    catalog = profiler.load(base / "catalog.json")
    return wh, catalog, meta


def _has_model() -> bool:
    return bool(os.getenv("QWEN_API_KEY") or os.getenv("QWEN_API_URL"))


# --- pages ------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html",
                        headers={"Cache-Control": "no-cache"})


# --- upload -----------------------------------------------------------------
@app.post("/api/datasets")
async def create_dataset(files: list[UploadFile], name: str = "") -> dict:
    """Ingest one or more uploads into a fresh DuckDB warehouse, then profile it.

    Multiple files land in the same warehouse on purpose — an analyst uploading
    orders.csv and customers.csv wants to join them, and the profiler's overlap
    scan will find the key that lets them."""
    if not files:
        raise HTTPException(400, "no files uploaded")

    dataset_id = uuid.uuid4().hex[:16]
    base = _dir(dataset_id)
    uploads = base / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    total = 0
    try:
        for f in files:
            stem = Path(f.filename or "upload").name
            ext = Path(stem).suffix.lower()
            if ext not in SUPPORTED:
                raise HTTPException(
                    400, f"{stem}: unsupported file type. Upload "
                         f"{', '.join(sorted(SUPPORTED))}.")
            target = uploads / stem
            with target.open("wb") as out:
                while chunk := await f.read(CHUNK):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413, f"upload exceeds "
                                 f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                    out.write(chunk)
            saved.append(target)

        try:
            tables = await asyncio.to_thread(ingest, saved,
                                             base / "warehouse.duckdb")
        except IngestError as e:
            raise HTTPException(400, f"could not read the upload: {e}")

        wh = Warehouse(base / "warehouse.duckdb")
        sources = {t.name: t.source_file for t in tables}
        label = name.strip() or (saved[0].stem if len(saved) == 1
                                 else f"{len(saved)} files")
        catalog = await asyncio.to_thread(
            profiler.build_catalog, wh, dataset_id, label, sources,
            QwenClient() if _has_model() else None, _has_model())
        profiler.save(catalog, base / "catalog.json")

        starters = await asyncio.to_thread(_starters, catalog)
        meta = {"id": dataset_id, "name": label, "created": time.time(),
                "files": [p.name for p in saved],
                "tables": [t.name for t in tables],
                "rows": sum(t.rows for t in tables),
                "bytes": total, "starters": starters}
        _write_meta(dataset_id, meta)
        return {"dataset": meta, "catalog": catalog.to_dict()}
    except HTTPException:
        shutil.rmtree(base, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(base, ignore_errors=True)
        raise HTTPException(500, f"{type(e).__name__}: {e}")


def _starters(catalog) -> list[str]:
    """Opening questions come from the uploaded schema — there is no canned list
    to fall back on, only a schema-derived one."""
    if _has_model():
        try:
            agent = SuggestAgent(QwenClient(), Budget(), Metrics(),
                                 lambda e: None)
            questions = agent.run(catalog, n=6)
            if questions:
                return questions
        except Exception:
            pass
    return suggest_offline(catalog)


# --- dataset reads ----------------------------------------------------------
@app.get("/api/datasets")
def list_datasets() -> dict:
    if not DATA_ROOT.exists():
        return {"datasets": [], "model_configured": _has_model()}
    out = []
    for path in DATA_ROOT.iterdir():
        meta = path / "meta.json"
        if meta.is_file():
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    out.sort(key=lambda d: d.get("created", 0), reverse=True)
    return {"datasets": out, "model_configured": _has_model()}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    meta = _read_meta(dataset_id)
    catalog = profiler.load(_dir(dataset_id) / "catalog.json")
    return {"dataset": meta, "catalog": catalog.to_dict()}


@app.get("/api/datasets/{dataset_id}/preview")
def preview(dataset_id: str, table: str, limit: int = PREVIEW_ROWS) -> dict:
    """First rows of one table, for the schema explorer."""
    wh, catalog, _ = _open(dataset_id)
    if not catalog.table(table):
        raise HTTPException(404, f"no table named {table!r}")
    limit = max(1, min(limit, 200))
    real = catalog.table(table).name
    rows = wh.introspect(f'SELECT * FROM "{real}" LIMIT {limit}')
    cols = wh.columns(real)
    from warehouse import _jsonable
    return {"table": real, "columns": [c for c, _ in cols],
            "types": [t for _, t in cols],
            "rows": [[_jsonable(v) for v in r] for r in rows]}


@app.get("/api/datasets/{dataset_id}/export")
def export(dataset_id: str, sql: str):
    """CSV of a result. The same guard the agents run against applies here —
    the query string is user-supplied, so it is not trusted."""
    wh, _, meta = _open(dataset_id)
    try:
        result = wh.query(guard(sql))
    except QueryError as e:
        raise HTTPException(400, str(e))

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(result.columns)
    writer.writerows(result.rows)
    filename = f"{meta.get('name', 'result')}.csv".replace(" ", "_")
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict:
    base = _dir(dataset_id)
    if not base.exists():
        raise HTTPException(404, "dataset not found")
    shutil.rmtree(base, ignore_errors=True)
    return {"deleted": dataset_id}


# --- one turn of the conversation -------------------------------------------
NDJSON = "application/x-ndjson"


def _line(event: dict) -> str:
    return json.dumps(event, default=str) + "\n"


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    def fail(message: str) -> StreamingResponse:
        async def err():
            yield _line({"type": "error", "message": message})
            yield _line({"type": "done"})
        return StreamingResponse(err(), media_type=NDJSON)

    message = body.message.strip()
    if not _has_model():
        return fail("No model configured. Set QWEN_API_URL and QWEN_API_KEY in "
                    ".env, then restart the server.")
    if not message:
        return fail("Say something about the data.")

    try:
        wh, catalog, _ = _open(body.dataset)
    except HTTPException as e:
        return fail(str(e.detail))

    history = [Exchange(question=t.question, answer=t.answer, sql=t.sql,
                        columns=list(t.columns))
               for t in body.history[-HISTORY_TURNS:] if t.question]
    max_context = max(4000, min(body.max_context, 200000))
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work() -> None:
        try:
            ChatPipeline(wh, catalog, budget=Budget(max_context=max_context),
                         max_repairs=body.max_repairs).run(message, history,
                                                           on_event=emit)
        except Exception as exc:
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            emit({"type": "done"})
        finally:
            emit({"type": "__end__"})

    loop.run_in_executor(None, work)

    async def stream():
        while True:
            event = await queue.get()
            if event.get("type") == "__end__":
                break
            yield _line(event)
            if await request.is_disconnected():
                break

    return StreamingResponse(stream(), media_type=NDJSON,
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
