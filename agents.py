"""The agent roster.

Five narrow specialists, each with its own system prompt, its own slice of
context and its own budgeted call:

    Router     message + conversation + whole schema -> answer it now, or
                                                       restate it as a
                                                       standalone question and
                                                       name the tables
    SQLAuthor  question + focused schema + last query -> one DuckDB SELECT
    SQLCritic  failed SQL + the exact error           -> repaired SQL (bounded)
    ChartAgent question + result shape                -> SHOULD there be a
                                                         graph, and which one
    Narrator   question + the rows that came back     -> the reply
    Suggest    schema + what just ran                 -> what to ask next

The Router is what makes this a conversation rather than a form: it resolves
"and by zone?" against the turn before it, and it answers outright when the
schema already holds the answer — nobody needs a SQL query to be told what
columns a file has.

Only the ChartAgent and the Narrator ever see result rows; only the Router sees
the whole catalog. That is what keeps every call inside a small window.

Two honesty rules survive from the grounded build, because they cost nothing:
the Router has seen the schema and never the values, so it is forbidden from
stating a figure; and the Narrator is given the rows and told that every number
it writes has to come from them.
"""

from __future__ import annotations

import re
import textwrap
from typing import Callable

import context as ctx
from context import Budget
from llm import QwenClient
from models import (Catalog, ChartSpec, Exchange, Metrics, QueryResult, Route)
from profiler import is_numeric, is_temporal

Emit = Callable[[dict], None]

# Output reserves per call — what we leave in the window for the reply.
ROUTE_OUTPUT = 700
SQL_OUTPUT = 700
REPAIR_OUTPUT = 700
CHART_OUTPUT = 500
NARRATE_OUTPUT = 900
SUGGEST_OUTPUT = 350

HISTORY_TURNS = 6       # how much of the conversation the router reads back

MAX_SERIES = 8          # categorical token ceiling
MAX_ALL_PAIRS_SERIES = 3   # scatter/bubble: every pair is adjacent, so cap lower
MAX_CATEGORIES = 30     # bars past this fold into a labelled tail


class Agent:
    """Shared plumbing: budgeted, metered, streamed structured calls."""

    name = "agent"

    def __init__(self, client: QwenClient, budget: Budget, metrics: Metrics,
                 emit: Emit):
        self.client = client
        self.budget = budget
        self.m = metrics
        self.emit = emit

    def _call(self, system: str, user: str, schema: dict, output_reserve: int,
              temperature: float = 0.1) -> dict:
        prompt_tokens = ctx.estimate_tokens(system) + ctx.estimate_tokens(user)
        self.m.context_peak = max(self.m.context_peak,
                                  prompt_tokens + output_reserve)
        self.emit({"type": "context", "agent": self.name,
                   "prompt_tokens": prompt_tokens,
                   "output_reserve": output_reserve,
                   "call_tokens": prompt_tokens + output_reserve,
                   "max_context": self.budget.max_context})
        data, text = self.client.structured(system, user, schema,
                                            output_reserve, temperature)
        self.m.calls += 1
        self.m.input_tokens += prompt_tokens
        self.m.output_tokens += ctx.estimate_tokens(text)
        return data

    def _overhead(self, system: str, *fixed: str) -> int:
        return ctx.estimate_tokens(system) + sum(ctx.estimate_tokens(f) for f in fixed)


# --- 1. Router --------------------------------------------------------------
ROUTER_SYSTEM = textwrap.dedent("""
    You are the front of a chat assistant that talks about ONE dataset someone
    uploaded. You see the derived schema — every table, its columns, their
    measured types, null rates and real sample values — the conversation so far,
    and the newest message. Route that message.

    mode "data" — answering needs the actual values: a total, a count, an
      average, a ranking, a trend, a breakdown, a filter, "show me", "how many",
      "which one", "is it up or down", anything about what the numbers say. When
      you are unsure, choose "data": looking beats guessing.
    mode "chat" — the schema and the conversation already answer it: a greeting
      or a thanks, "what is in this file?", "which columns are dates?", "what
      does zone mean?", "what can you ask this data?", "what did you just run?",
      or a question this dataset simply cannot answer.

    When mode is "data":
    - Write `question`: the newest message rewritten as ONE self-contained
      question, with every reference resolved from the conversation. After a
      question about total GTV, "and by zone?" becomes "What is the total GTV by
      zone?" and "same for June" becomes the earlier question with June in it.
      If the message already stands alone, copy it as it is.
    - Put in `tables` only the tables the answer needs, spelled exactly as the
      schema spells them.
    - Leave `reply` empty. Do not answer — you have not seen a single row.

    When mode is "chat":
    - Write `reply` yourself. Warm, direct, two or three sentences. No headings,
      and no dump of the whole schema unless they asked for the columns.
    - You have seen the SCHEMA, not the DATA. It tells you which columns exist,
      what type they are and a few sample values — never a total, an average or
      a rank. NEVER state a figure as fact in `reply`. If a number would answer
      the message, then the message was mode "data".
    - If this dataset genuinely cannot answer what they asked, say plainly what
      is missing and name one thing it CAN answer instead. Never invent a column
      or a business rule to be helpful.
    - Leave `question` and `tables` empty.
""").strip()

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["chat", "data"]},
        "reply": {"type": "string"},
        "question": {"type": "string"},
        "tables": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["mode"],
}


def history_block(history: list[Exchange], turns: int = HISTORY_TURNS) -> str:
    recent = [h for h in (history or []) if h.question][-turns:]
    if not recent:
        return ""
    return ("Conversation so far:\n"
            + "\n".join(h.compact() for h in recent) + "\n\n")


class RouterAgent(Agent):
    name = "router"

    def run(self, message: str, catalog: Catalog,
            history: list[Exchange] | None = None) -> Route:
        fixed = (f"Dataset: {catalog.name}\n{catalog.summary}\n\n"
                 f"{history_block(history)}"
                 f"Newest message: {message}\n\nSchema:\n")
        room = self.budget.room_for(ROUTE_OUTPUT,
                                    self._overhead(ROUTER_SYSTEM, fixed))
        schema_text, full, total = ctx.fit_schema(catalog, room)
        if full < total:
            self.emit({"type": "note", "agent": self.name,
                       "message": f"Context budget: {full}/{total} tables shown in "
                                  f"full detail, the rest as name + columns."})

        data = self._call(ROUTER_SYSTEM, fixed + schema_text, ROUTE_SCHEMA,
                          ROUTE_OUTPUT, temperature=0.2)
        route = Route(
            mode=str(data.get("mode", "data")).strip().lower(),
            reply=str(data.get("reply", "")).strip(),
            question=str(data.get("question", "")).strip() or message,
            tables=_strings(data.get("tables")),
        )
        return _ground_route(route, catalog)


def _ground_route(route: Route, catalog: Catalog) -> Route:
    """Hold the router to what exists.

    An unknown mode, or a "chat" with nothing in it, falls through to "data" —
    an empty bubble is the one reply a chatbot must never send. Tables are
    filtered to real ones, and the quality flags the profiler already measured
    for them come along, so the SQL author does not have to rediscover a
    duplicate-rows problem the statistics proved at upload."""
    if route.mode not in ("chat", "data"):
        route.mode = "data"
    if route.mode == "chat" and not route.reply:
        route.mode = "data"

    real = {t.name.lower(): t.name for t in catalog.tables}
    route.tables = list(dict.fromkeys(real[t.lower()] for t in route.tables
                                      if t.lower() in real))
    if route.mode == "data" and not route.tables and catalog.tables:
        route.tables = [t.name for t in catalog.tables]

    touched = {t.lower() for t in route.tables}
    for q in catalog.quality:
        if q.table.lower() in touched and q.severity == "serious":
            note = f"{q.table}.{q.column}: {q.detail}"
            if note not in route.caveats:
                route.caveats.append(note)
    return route


# --- 2. SQL author ----------------------------------------------------------
SQL_SYSTEM = textwrap.dedent("""
    You write DuckDB SQL. You are given one question and the schema of the
    tables that answer it. Return ONE read-only SELECT statement.

    Rules:
    - DuckDB dialect. One statement, no semicolon, no comments. SELECT or WITH
      only — never INSERT, UPDATE, CREATE, ATTACH, COPY or any file-reading
      function.
    - Use only tables and columns from the schema, spelled exactly as given.
      Double-quote any identifier that is not plain lowercase.
    - Alias every computed column to a short, readable snake_case name. Those
      names are what the reader sees on the chart, so make them mean something.
    - Answer at the grain the question asks for and nothing wider. Aggregate
      rather than dumping rows. ORDER BY whatever makes the answer readable, and
      LIMIT when the question asks for a top or bottom N.
    - Return a shape a person can read: a handful of grouped rows beats four
      thousand raw ones. If the question implies a comparison, put the things
      being compared side by side.
    - Guard the arithmetic: divide with NULLIF(denominator, 0), and cast text
      that holds numbers with TRY_CAST before doing maths on it.
    - If a caveat flags duplicate rows, de-duplicate before aggregating.
    - This is a conversation. If the previous query is shown and the question is
      a variation on it, keep its filters, aliases and definitions and change
      only what was asked — a follow-up should not silently redefine the metric.
""").strip()

SQL_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["sql"],
}

_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _clean_sql(sql: str) -> str:
    return _FENCE.sub("", (sql or "").strip()).strip().rstrip(";").strip()


class SQLAuthorAgent(Agent):
    name = "sql-author"

    def run(self, question: str, route: Route, catalog: Catalog,
            previous_sql: str = "") -> tuple[str, str]:
        fixed = f"Question: {question}\n"
        if previous_sql:
            fixed += (f"\nThe query that answered the previous message:\n"
                      f"{previous_sql}\n")
        if route.caveats:
            fixed += ("\nMeasured caveats to handle in SQL:\n"
                      + "\n".join(f"  - {c}" for c in route.caveats) + "\n")
        fixed += "\nSchema:\n"
        room = self.budget.room_for(SQL_OUTPUT, self._overhead(SQL_SYSTEM, fixed))
        schema_text = ctx.focus_schema(catalog, route.tables, room)

        data = self._call(SQL_SYSTEM, fixed + schema_text, SQL_SCHEMA, SQL_OUTPUT)
        return _clean_sql(data.get("sql", "")), str(data.get("explanation", "")).strip()


# --- 3. SQL critic / repair -------------------------------------------------
REPAIR_SYSTEM = textwrap.dedent("""
    You repair DuckDB SQL. You get a query, the exact error the database
    returned, and the schema of the tables involved. Return a corrected query.

    Rules:
    - Fix the actual cause. An unknown-column error means the name is wrong —
      find the right one in the schema; do not delete the column and change what
      the query means. A type error means add TRY_CAST, not drop the comparison.
    - Keep the query answering the same question at the same grain.
    - Still one read-only DuckDB SELECT, no semicolon.
    - In `diagnosis`, say in one sentence what was wrong.
""").strip()

REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}, "diagnosis": {"type": "string"}},
    "required": ["sql", "diagnosis"],
}


class SQLCriticAgent(Agent):
    name = "sql-critic"

    def run(self, question: str, route: Route, catalog: Catalog, sql: str,
            error: str) -> tuple[str, str]:
        fixed = (f"Question: {question}\n\n"
                 f"Query that failed:\n{sql}\n\n"
                 f"DuckDB said:\n{error}\n\nSchema:\n")
        room = self.budget.room_for(REPAIR_OUTPUT,
                                    self._overhead(REPAIR_SYSTEM, fixed))
        schema_text = ctx.focus_schema(catalog, route.tables, room)

        data = self._call(REPAIR_SYSTEM, fixed + schema_text, REPAIR_SCHEMA,
                          REPAIR_OUTPUT)
        return _clean_sql(data.get("sql", "")), str(data.get("diagnosis", "")).strip()


# --- 4. Chart agent — decides IF a graph belongs here at all ----------------
CHART_SYSTEM = textwrap.dedent("""
    You are the visualization analyst. You see the user's question, the SQL that
    answered it, and the shape of the result — column names, types, row count and
    the first rows. Decide whether a GRAPH would tell the reader something the
    table does not, and if so, which one.

    Say NO to a chart when:
    - The result is a single number or a single row. That is a stat tile
      (form "stat"), not a one-bar bar chart.
    - The result is a lookup — the reader wanted specific values, not a shape.
    - There is no measure to plot, only identifiers or free text (form "table").
    - Every category has a near-identical value, so the chart would show a flat
      line and imply a pattern that is not there.
    - The rows are unordered detail with no dimension worth comparing across.

    Say YES when a shape carries the answer: a trend over time, a ranking, a
    comparison across categories, a part-to-whole split, or a relationship
    between two measures.

    Pick the form by the reader's job:
      line / area   change over time (x must be the temporal column)
      bar           compare magnitude across few, short-named categories
      hbar          same, but many categories or long names
      grouped_bar   compare a measure across categories AND a second dimension
      stacked_bar   part-to-whole across categories, keeping the totals
      share         part-to-whole as PERCENTAGES, drawn as a pie — for "what is
                    the mix / the split / the share of total"
      scatter       relationship between two measures, one point per row
      stat          one headline number
      table         the values themselves are the answer

    Rules:
    - x, y, series and y_columns MUST be column names copied exactly from the
      result columns. Never invent one, never rename one.
    - y must be a numeric column. x is the dimension or the time column.
    - Set series only for a real second dimension; leave it empty otherwise.
    - WIDE RESULTS. Very often the things being compared are separate numeric
      COLUMNS rather than values in one dimension column — `may_2025_gtv` and
      `jun_2025_gtv`, or `revenue_2024` and `revenue_2025`. There is no series
      column to point at, so put EVERY one of those columns in y_columns and
      choose grouped_bar (or line, if they form a sequence over time). They are
      series of one chart, not separate charts. Leave y empty when you do this.
      Setting y to just one of them silently throws the comparison away.
    - Only put columns that share a UNIT and a SCALE in y_columns. A percentage
      change and a currency total do not belong on one axis — pick the group
      that answers the question and leave the other out.
    - share vs stacked_bar. stacked_bar keeps the magnitudes, so it answers "how
      big is each, and how big together". share throws the magnitudes away and
      normalises every whole to 100%, so it answers "what is the mix". Choose
      share when the question asks for a share, a mix, a split, a proportion, a
      percentage of total, or a composition — and never when any value being
      plotted is negative, because a percentage of a total that crosses zero is
      not a real quantity. A share of one dimension only (no second dimension)
      is perfectly valid: it draws as one bar split into the categories.
    - color_job: "categorical" only when distinct series ARE the subject;
      "sequential" for a single measure; "diverging" when values cross a
      meaningful zero (profit/loss, change vs target).
    - Set emphasis to one x-value when the question is about that one item and
      the rest are context.
    - Write reason as one sentence a reader would find useful — say what the
      chart shows, or why the table is already the better answer.
""").strip()

CHART_SCHEMA = {
    "type": "object",
    "properties": {
        "should_chart": {"type": "boolean"},
        "reason": {"type": "string"},
        "form": {"type": "string",
                 "enum": ["line", "area", "bar", "hbar", "grouped_bar",
                          "stacked_bar", "share", "scatter", "stat", "table"]},
        "x": {"type": "string"},
        "y": {"type": "string"},
        "series": {"type": "string"},
        "y_columns": {"type": "array", "items": {"type": "string"}},
        "color_job": {"type": "string",
                      "enum": ["sequential", "categorical", "diverging", "none"]},
        "emphasis": {"type": "string"},
        "title": {"type": "string"},
        "x_label": {"type": "string"},
        "y_label": {"type": "string"},
        "value_format": {"type": "string",
                         "enum": ["number", "compact", "currency", "percent"]},
        "sort": {"type": "string",
                 "enum": ["none", "value_desc", "value_asc", "x_asc"]},
    },
    "required": ["should_chart", "reason", "form"],
}


class ChartAgent(Agent):
    name = "chart"

    def run(self, question: str, sql: str, result: QueryResult) -> ChartSpec:
        if not result.rows:
            return ChartSpec(False, "The query returned no rows, so there is "
                                    "nothing to plot.", form="table")

        room = self.budget.room_for(CHART_OUTPUT, self._overhead(CHART_SYSTEM))
        preview = ctx.fit_rows(result, max(200, room - 400), head=12, tail=3)
        user = (f"Question: {question}\n\nSQL:\n{sql}\n\n"
                f"Result: {result.row_count:,} rows\n"
                f"Columns: {result.schema_line()}\n\nFirst rows:\n{preview}")

        data = self._call(CHART_SYSTEM, user, CHART_SCHEMA, CHART_OUTPUT,
                          temperature=0.0)
        spec = ChartSpec(
            should_chart=bool(data.get("should_chart")),
            reason=str(data.get("reason", "")).strip(),
            form=str(data.get("form", "table")).strip().lower(),
            x=str(data.get("x", "")).strip(),
            y=str(data.get("y", "")).strip(),
            series=str(data.get("series", "")).strip(),
            y_columns=_strings(data.get("y_columns")),
            color_job=str(data.get("color_job", "sequential")).strip().lower(),
            emphasis=str(data.get("emphasis", "")).strip(),
            title=str(data.get("title", "")).strip(),
            x_label=str(data.get("x_label", "")).strip(),
            y_label=str(data.get("y_label", "")).strip(),
            value_format=str(data.get("value_format", "number")).strip().lower(),
            sort=str(data.get("sort", "none")).strip().lower(),
        )
        return reconcile_chart(spec, result)


SCALE_BAND = 50.0   # how far below the biggest measure a column may sit


def _same_scale(columns: list[str], result: QueryResult) -> list[str]:
    """Keep only the columns that can share one value axis.

    A grouped bar of `jun_gtv` (7.6e9), `may_gtv` (3.1e9) and `gtv_pct_change`
    (0.06) renders the percentage as an invisible sliver against the baseline
    and reads as "the change is zero". The alternative — a second y-axis — is
    the single worst chart mistake there is, so the odd column is dropped and
    the reader is told, rather than quietly given a lie."""
    if len(columns) < 2:
        return list(columns)
    peaks: dict[str, float] = {}
    for c in columns:
        i = result.columns.index(c)
        vals = [abs(r[i]) for r in result.rows
                if isinstance(r[i], (int, float)) and r[i] is not None]
        peaks[c] = max(vals) if vals else 0.0
    reference = max(peaks.values(), default=0.0)
    if reference <= 0:
        return list(columns)
    return [c for c in columns if peaks[c] * SCALE_BAND >= reference]


def reconcile_chart(spec: ChartSpec, result: QueryResult) -> ChartSpec:
    """Check the agent's spec against the data it claims to describe.

    A model can name a column that isn't in the result, put a text column on a
    value axis, or ask for a bar per row across 4,000 rows. This pass is
    deterministic: it repairs what it can, vetoes what it can't, and records
    every change in `notes` so the UI can show what was overridden."""
    cols = list(result.columns)
    types = {c: t for c, t in zip(result.columns, result.types)}
    lower = {c.lower(): c for c in cols}
    notes: list[str] = []

    def resolve(name: str) -> str:
        if not name:
            return ""
        if name in lower.values():
            return name
        return lower.get(name.lower(), "")

    numeric = [c for c in cols if is_numeric(types[c])]
    temporal = [c for c in cols if is_temporal(types[c])]
    categorical = [c for c in cols if c not in numeric]

    # 0 rows, or a single scalar: never a graph.
    if result.row_count == 0:
        return ChartSpec(False, "The query returned no rows.", form="table")
    if result.row_count == 1 and len(numeric) >= 1 and len(cols) <= 3:
        return ChartSpec(
            False,
            spec.reason or "The answer is a single number — a stat tile reads it "
                           "faster than a one-bar chart.",
            form="stat", y=resolve(spec.y) or numeric[0],
            x=resolve(spec.x) or (categorical[0] if categorical else ""),
            title=spec.title, value_format=spec.value_format,
            notes=["Single-row result rendered as a stat tile."])

    if not spec.should_chart or spec.form in ("table", "stat", "none"):
        form = spec.form if spec.form in ("stat", "table") else "table"
        return ChartSpec(False, spec.reason or "The table is the clearer answer.",
                         form=form, title=spec.title,
                         value_format=spec.value_format)

    x, y, series = resolve(spec.x), resolve(spec.y), resolve(spec.series)

    # --- wide results: several measure columns are several series ----------
    ys = list(dict.fromkeys(c for c in (resolve(n) for n in spec.y_columns)
                            if c in numeric))
    autodetected = False
    if len(ys) < 2 and not series and spec.form in (
            "bar", "hbar", "grouped_bar", "stacked_bar", "line", "area"):
        # Safety net. Asked to "compare June to May by zone", the agent reliably
        # describes a grouped bar and then names a single column, silently
        # dropping half the comparison. If the result is one dimension plus
        # several same-scale measures, that IS a multi-series chart.
        candidates = _same_scale([c for c in numeric if c != x], result)
        if len(candidates) >= 2:
            ys, autodetected = candidates, True
    if len(ys) >= 2:
        kept = _same_scale(ys, result)
        if len(kept) < len(ys):
            dropped = [c for c in ys if c not in kept]
            notes.append(f"{', '.join(dropped)} sits on a different scale and "
                         f"would flatten the others — left off this axis.")
        ys = kept

    if len(ys) > MAX_SERIES:
        notes.append(f"{len(ys)} measures is past the {MAX_SERIES}-colour "
                     f"ceiling — showing the first {MAX_SERIES}.")
        ys = ys[:MAX_SERIES]

    if len(ys) >= 2:
        if autodetected:
            notes.append(f"Result is wide: {', '.join(ys)} are separate columns "
                         f"measuring the same thing, so they are plotted as "
                         f"series of one chart rather than one of them alone.")
        if spec.form in ("bar", "hbar", "grouped_bar"):
            spec.form = "grouped_bar"
        spec.color_job = "categorical"
        spec.emphasis = ""            # identity is the point; nothing recedes
        y = ys[0]
        series = ""
    else:
        ys = []
    spec.y_columns = ys

    # A value axis needs a number. Repair from the result if we can; veto if not.
    if y not in numeric:
        if not numeric:
            return ChartSpec(False, "No numeric column came back, so there is "
                                    "nothing to plot on a value axis.",
                             form="table")
        notes.append(f'y "{spec.y or "(unset)"}" is not numeric — plotting '
                     f'"{numeric[0]}" instead.')
        y = numeric[0]

    if not x:
        pick = (temporal or [c for c in categorical if c != y]
                or [c for c in numeric if c != y])
        if not pick:
            return ChartSpec(False, "The result has only one column, so there is "
                                    "no dimension to plot it against.",
                             form="stat", y=y)
        x = pick[0]
        notes.append(f'x was unset — using "{x}".')
    if x == y and len(cols) > 1:
        alt = next((c for c in cols if c != y), "")
        if alt:
            notes.append(f'x and y were the same column — using "{alt}" for x.')
            x = alt

    form = spec.form
    if form in ("line", "area") and x not in temporal and x not in numeric:
        # A line needs an ordered axis, but the axis does not have to be a DATE.
        # Text-to-SQL routinely returns months as strings ('2025-01'), and one
        # row per label is exactly the sequence a line chart wants.
        xi_ = cols.index(x)
        labels = [r[xi_] for r in result.rows]
        if temporal:
            notes.append(f'{form} needs an ordered axis — using "{temporal[0]}".')
            x = temporal[0]
        elif len(set(labels)) == len(labels) and len(labels) > 2:
            notes.append(f'"{x}" is text rather than a date, but holds one '
                         f'ordered value per row — kept as a {form} in the '
                         f'order the query returned.')
        else:
            notes.append(f"{form} implies an ordered axis but \"{x}\" repeats "
                         f"and no date column came back — switched to a bar "
                         f"chart.")
            form = "bar"

    if form == "scatter":
        if x not in numeric:
            others = [c for c in numeric if c != y]
            if others:
                notes.append(f'scatter needs two measures — using "{others[0]}" '
                             f'for x.')
                x = others[0]
            else:
                notes.append("scatter needs two numeric columns — switched to a "
                             "bar chart.")
                form = "bar"

    if form == "share":
        # A share is only arithmetic worth drawing when every part sits on the
        # same side of zero. Let a negative in and the parts can exceed the
        # whole, or the whole lands near zero and every share explodes.
        plotted = ys or [y]
        negative = any(
            isinstance(r[cols.index(c)], (int, float)) and r[cols.index(c)] < 0
            for c in plotted for r in result.rows)
        if negative:
            notes.append("Share needs parts of one positive whole and this result "
                         "has negative values — drawn as a bar chart instead, "
                         "where a negative still means something.")
            form = "bar"
        else:
            # The segments ARE the subject here, whether they are series or the
            # categories of a single measure.
            spec.color_job = "categorical"
            spec.emphasis = ""

    if series and series in (x, y):
        notes.append("series duplicated another encoding — dropped.")
        series = ""
    if form in ("grouped_bar", "stacked_bar") and not series and not ys:
        notes.append(f"{form} needs a second dimension — switched to a bar chart.")
        form = "bar"

    # Series-count ceilings (see the categorical ladder: 8 for adjacent forms,
    # 3 where every pair sits side by side).
    if series:
        si = cols.index(series)
        distinct = {r[si] for r in result.rows if r[si] is not None}
        cap = MAX_ALL_PAIRS_SERIES if form == "scatter" else MAX_SERIES
        if len(distinct) > cap:
            if form == "scatter":
                notes.append(f'{len(distinct)} groups is past the {cap}-series cap '
                             f'for a scatter — dropped the colour split.')
                series = ""
            else:
                notes.append(f'{len(distinct)} series is past the {cap}-colour '
                             f'ceiling — the tail is folded into "Other".')
                spec.limit = spec.limit or 0

    # Too many bars to read: rank and keep the head, and say so.
    xi = cols.index(x) if x in cols else -1
    if form in ("bar", "hbar", "stacked_bar", "grouped_bar") and xi >= 0:
        categories = len({r[xi] for r in result.rows})
        if categories > MAX_CATEGORIES:
            spec.limit = MAX_CATEGORIES
            if spec.sort == "none":
                spec.sort = "value_desc"
            notes.append(f"{categories:,} categories — showing the top "
                         f"{MAX_CATEGORIES} by value.")
        if categories > 12 and form == "bar":
            longest = max((len(str(r[xi])) for r in result.rows), default=0)
            if longest > 10:
                form = "hbar"
                notes.append("Long category names — laid out horizontally.")

    # A lone series normally wants one hue — but a single-measure share bar is
    # split into its categories, and those slices need telling apart.
    if not series and not ys and spec.color_job == "categorical" and form != "share":
        spec.color_job = "sequential"
        notes.append("One series — using a single hue rather than a categorical "
                     "palette.")
    if spec.color_job == "diverging":
        yi = cols.index(y)
        vals = [r[yi] for r in result.rows if isinstance(r[yi], (int, float))]
        if not (vals and min(vals) < 0 < max(vals)):
            spec.color_job = "sequential"
            notes.append("Values never cross zero — a diverging palette would "
                         "imply a polarity the data does not have.")

    if spec.emphasis and xi >= 0:
        seen = {str(r[xi]) for r in result.rows}
        if spec.emphasis not in seen:
            notes.append(f'emphasis "{spec.emphasis}" is not in the result — '
                         f'dropped.')
            spec.emphasis = ""

    spec.should_chart = True
    spec.form, spec.x, spec.y, spec.series = form, x, y, series
    spec.x_label = spec.x_label or x
    spec.y_label = spec.y_label or y
    spec.title = spec.title or f"{y} by {x}"
    spec.notes = notes
    return spec


# --- 5. Narrator ------------------------------------------------------------
NARRATOR_SYSTEM = textwrap.dedent("""
    You are an analyst in the middle of a chat. You get the question that was
    asked and the ACTUAL rows that came back. Reply to the person.

    Rules:
    - Lead with the direct answer in the first sentence, with the number in it.
      Then at most two more short sentences on what the numbers show — the
      shape, the outlier, the comparison that matters. Stop there. This is a
      chat message, not a report: no headings, no preamble, no sign-off, and no
      bullet list unless you are naming three or more items.
    - Every number you write must come from the rows you were given. Quote it to
      at most two decimal places — 324676.71000000014 is floating-point noise and
      should be written 324676.71 — but never restate a figure as a different
      number, never estimate, and never add a number the query did not return.
      There is no number you are allowed to work out in your head.
    - CHECK THE COLUMN HEADER OF EVERY FIGURE YOU QUOTE. When the result has
      more than one measure column — may_revenue next to june_revenue,
      revenue_2024 next to revenue_2025 — each number belongs to exactly one of
      them, and a clause that names one column while quoting the other's number
      is wrong even though both numbers are real. Attach each figure to its own
      column in its own clause: "North rose from 4,310.60 in May to 71,980.00 in
      June", never "North led in May with 71,980.00 in June".
    - A chart is being drawn from these same rows, so do not describe it, and do
      not read the whole table out loud.
    - Never mention SQL, queries, tables or columns-as-columns. The reader asked
      a question in English and wants it answered in English.
    - If no rows came back, say so plainly in one sentence and say what would be
      worth trying instead.
    - If a caveat means this answer is shakier than it looks, say it in one short
      closing sentence. Otherwise leave it out.
""").strip()

NARRATE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class NarratorAgent(Agent):
    name = "narrator"

    def run(self, question: str, route: Route, result: QueryResult) -> str:
        fixed = (f"Question: {question}\n"
                 + ("Caveats:\n" + "\n".join(f"  - {c}" for c in route.caveats)
                    + "\n" if route.caveats else "")
                 + f"\nRows returned: {result.row_count:,}\n\nResult:\n")
        room = self.budget.room_for(NARRATE_OUTPUT,
                                    self._overhead(NARRATOR_SYSTEM, fixed))
        rows = ctx.fit_rows(result, room)

        data = self._call(NARRATOR_SYSTEM, fixed + rows, NARRATE_SCHEMA,
                          NARRATE_OUTPUT, temperature=0.2)
        return str(data.get("answer", "")).strip()


# --- 6. Suggester -----------------------------------------------------------
SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}


class SuggestAgent(Agent):
    """What to ask next. Runs cold right after an upload — which is why the
    landing page needs no canned examples, the chips come from the file that was
    just dropped — and again after a data turn, where it can also see what came
    back."""

    name = "suggest"

    def run(self, catalog: Catalog, asked: str = "",
            result: QueryResult | None = None, n: int = 3) -> list[str]:
        system = (
            f"You suggest the next question to ask about a dataset. Propose {n} "
            "short questions this data can genuinely answer — each reachable "
            "from the columns shown, each opening a different direction (a "
            "total, a ranking, a trend over time if a date column exists, a "
            "breakdown by category, a comparison). Phrase them the way someone "
            "would type them into a chat, under 12 words each. Never mention a "
            "column the schema does not contain."
        )
        head = f"Dataset: {catalog.name}\n{catalog.summary}\n"
        if asked:
            head += (f"\nJust answered: {asked}\n"
                     "Do not repeat that question or a rewording of it.\n")
        if result is not None:
            head += (f"It returned {result.row_count:,} rows: "
                     f"{result.schema_line()}\n")
        head += "\nSchema:\n"
        room = self.budget.room_for(SUGGEST_OUTPUT, self._overhead(system, head))
        schema_text, _, _ = ctx.fit_schema(catalog, room)
        data = self._call(system, head + schema_text, SUGGEST_SCHEMA,
                          SUGGEST_OUTPUT, temperature=0.5)
        return _strings(data.get("questions"))[:n]


def _strings(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if isinstance(x, (str, int, float))
            and str(x).strip()]


def suggest_offline(catalog: Catalog, n: int = 6) -> list[str]:
    """Schema-derived starters that need no LLM — used when the model is
    unreachable, so an upload is never a dead end."""
    out: list[str] = []
    for t in catalog.tables[:3]:
        measures = [c for c in t.columns if c.role == "measure"]
        dims = [c for c in t.columns if c.role == "dimension"]
        times = [c for c in t.columns if c.role == "temporal"]
        out.append(f"How many rows are in {t.name}?")
        if measures and dims:
            out.append(f"What is the total {measures[0].name} by "
                       f"{dims[0].name}?")
        if measures and times:
            out.append(f"How has {measures[0].name} changed over "
                       f"{times[0].name}?")
        if dims:
            out.append(f"Which {dims[0].name} appears most often in {t.name}?")
        if measures:
            out.append(f"What is the average {measures[0].name} in {t.name}?")
    seen, unique = set(), []
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique[:n]
