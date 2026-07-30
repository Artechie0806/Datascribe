"""One turn of the conversation.

    message + the conversation so far
       │
       ▼
    Router ──► mode "chat" ──► its reply IS the answer, stop (one call)
       │
       ▼ mode "data", with the message rewritten to stand alone
    SQL author ──► guard ──► EXPLAIN ──► execute
       │                        │
       │                        └── error ──► SQL critic ──► retry (bounded)
       ▼
    result rows
       ├──► Chart agent   (should a graph exist here? which one?)
       └──► Narrator ──► the reply
                          │
                          ▼
                    Suggest ──► what to ask next

Small talk costs one call; a real question costs four. The Chart agent and the
Narrator run concurrently — both need the rows and neither needs the other.

`on_event` streams every step as a dict so the UI can render the turn as it
happens, and `history` is what makes it a conversation: the router reads it back
and the SQL author sees the query it wrote last time, so "and by zone?" means
something.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from agents import (ChartAgent, NarratorAgent, RouterAgent, SQLAuthorAgent,
                    SQLCriticAgent, SuggestAgent)
from context import Budget
from llm import QwenClient
from models import (Catalog, Exchange, Metrics, QueryResult, Route, SQLAttempt,
                    TurnResult)
from warehouse import QueryError, Warehouse

Emit = Callable[[dict], None]

MAX_REPAIRS = 3


class ChatPipeline:
    def __init__(self, warehouse: Warehouse, catalog: Catalog,
                 budget: Budget | None = None, client: QwenClient | None = None,
                 max_repairs: int = MAX_REPAIRS):
        self.wh = warehouse
        self.catalog = catalog
        self.budget = budget or Budget()
        self.client = client or QwenClient()
        self.max_repairs = max(0, min(max_repairs, 5))

    def _agents(self, m: Metrics, emit: Emit) -> dict:
        args = (self.client, self.budget, m, emit)
        return {
            "router": RouterAgent(*args),
            "author": SQLAuthorAgent(*args),
            "critic": SQLCriticAgent(*args),
            "chart": ChartAgent(*args),
            "narrator": NarratorAgent(*args),
            "suggest": SuggestAgent(*args),
        }

    # --- execution with bounded repair --------------------------------------
    def _execute(self, question: str, route: Route, sql: str, agents: dict,
                 m: Metrics,
                 emit: Emit) -> tuple[QueryResult | None, str, list[SQLAttempt]]:
        attempts: list[SQLAttempt] = []
        current = sql

        for n in range(self.max_repairs + 1):
            m.sql_attempts += 1
            emit({"type": "sql", "attempt": n + 1, "sql": current})
            try:
                # Plan first: EXPLAIN catches an unknown column or a bad join
                # without paying to scan the table.
                self.wh.explain(current)
                result = self.wh.query(current)
            except QueryError as e:
                error = str(e)
                attempts.append(SQLAttempt(n + 1, current, ok=False, error=error))
                emit({"type": "sql_error", "attempt": n + 1, "error": error})
                if n >= self.max_repairs:
                    return None, current, attempts
                m.repairs += 1
                emit({"type": "status", "stage": "repair", "attempt": n + 1})
                repaired, diagnosis = agents["critic"].run(
                    question, route, self.catalog, current, error)
                if not repaired or repaired.strip() == current.strip():
                    return None, current, attempts
                attempts[-1].critique = diagnosis
                emit({"type": "repair", "attempt": n + 1, "diagnosis": diagnosis,
                      "sql": repaired})
                current = repaired
                continue

            attempts.append(SQLAttempt(n + 1, current, ok=True))
            return result, current, attempts

        return None, current, attempts

    # --- one turn -----------------------------------------------------------
    def run(self, message: str, history: list[Exchange] | None = None,
            on_event: Emit | None = None) -> TurnResult:
        emit: Emit = on_event or (lambda e: None)
        history = history or []
        m = Metrics()
        started = time.perf_counter()
        agents = self._agents(m, emit)
        out = TurnResult(question=message, metrics=m)

        emit({"type": "start", "message": message, "dataset": self.catalog.name,
              "max_context": self.budget.max_context})

        # 1. route: answer it here, or go and get the rows
        emit({"type": "status", "stage": "route"})
        route = agents["router"].run(message, self.catalog, history)
        out.route = route
        emit({"type": "route", **route.to_dict()})

        if route.mode == "chat":
            out.answer = route.reply
            emit({"type": "answer", "answer": route.reply})
            return self._finish(out, m, started, emit)

        # The router may have rewritten the message; everything downstream —
        # including the chart title and the suggester — works off that version.
        question = route.question or message
        previous_sql = next((h.sql for h in reversed(history) if h.sql), "")

        # 2. SQL
        emit({"type": "status", "stage": "sql"})
        sql, explanation = agents["author"].run(question, route, self.catalog,
                                               previous_sql)
        if explanation:
            emit({"type": "note", "agent": "sql-author", "message": explanation})

        # 3. execute (with repair)
        emit({"type": "status", "stage": "execute"})
        result, final_sql, attempts = self._execute(question, route, sql, agents,
                                                    m, emit)
        out.sql, out.attempts, out.result = final_sql, attempts, result

        if result is None:
            last = attempts[-1].error if attempts else "the query failed"
            out.answer = (f"I could not get that query to run. The database "
                          f"said: {last}")
            emit({"type": "error", "message": out.answer, "sql": final_sql})
            return self._finish(out, m, started, emit)

        emit({"type": "result", "columns": result.columns, "types": result.types,
              "rows": result.rows, "row_count": result.row_count,
              "truncated": result.truncated, "elapsed_ms": result.elapsed_ms,
              "sql": final_sql})

        # 4. chart decision + reply, concurrently
        emit({"type": "status", "stage": "read"})
        with ThreadPoolExecutor(max_workers=2) as pool:
            chart_fut = pool.submit(agents["chart"].run, question, final_sql,
                                    result)
            narrate_fut = pool.submit(agents["narrator"].run, question, route,
                                      result)
            chart = _safe(chart_fut.result, None)
            answer = _safe(narrate_fut.result, "")

        out.answer = answer or "Here is what came back."
        emit({"type": "answer", "answer": out.answer})
        if chart is not None:
            out.chart = chart
            emit({"type": "chart", **chart.to_dict()})

        # 5. where to go next
        out.suggestions = _safe(
            lambda: agents["suggest"].run(self.catalog, asked=question,
                                          result=result), [])
        if out.suggestions:
            emit({"type": "suggestions", "questions": out.suggestions})

        return self._finish(out, m, started, emit)

    def _finish(self, out: TurnResult, m: Metrics, started: float,
                emit: Emit) -> TurnResult:
        m.elapsed_ms = int((time.perf_counter() - started) * 1000)
        emit({"type": "metrics", **m.to_dict()})
        emit({"type": "done"})
        return out


def _safe(fn, default):
    """Run a best-effort stage. A chart suggestion or a chip list failing must
    not take the answer down with it."""
    try:
        return fn()
    except Exception:
        return default
