"""Shared domain models for the data chat.

Nothing here describes a *particular* dataset. Every schema fact — table names,
column meanings, grain, join keys — is derived at upload time by the profiler and
carried in these objects. There is no pre-defined database and no hand-written
data dictionary anywhere in the project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# --- catalog (built by the profiler, per uploaded dataset) -------------------
@dataclass
class ColumnProfile:
    name: str
    sql_type: str
    role: str = "unknown"       # dimension | measure | temporal | identifier | text
    nulls: int = 0
    null_pct: float = 0.0
    distinct: int = 0
    is_unique: bool = False
    min: str | None = None
    max: str | None = None
    mean: float | None = None
    samples: list[str] = field(default_factory=list)
    description: str = ""       # derived by the semantic pass — never hand-written

    def to_dict(self) -> dict:
        return asdict(self)

    def compact(self) -> str:
        """One dense line for prompts — the schema-linking agents see this."""
        bits = [f"{self.name} {self.sql_type}", self.role]
        if self.is_unique:
            bits.append("unique")
        if self.distinct:
            bits.append(f"{self.distinct} distinct")
        if self.null_pct > 1:
            bits.append(f"{self.null_pct:.0f}% null")
        if self.min is not None and self.role in ("measure", "temporal"):
            bits.append(f"range {self.min}..{self.max}")
        if self.samples:
            bits.append("e.g. " + ", ".join(repr(s) for s in self.samples[:3]))
        if self.description:
            bits.append(f"— {self.description}")
        return "  - " + "; ".join(bits)


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    source_file: str = ""
    description: str = ""       # derived
    grain: str = ""             # derived: what one row represents

    def to_dict(self) -> dict:
        return {"name": self.name, "rows": self.rows,
                "source_file": self.source_file, "description": self.description,
                "grain": self.grain,
                "columns": [c.to_dict() for c in self.columns]}

    def compact(self, only: list[str] | None = None) -> str:
        cols = [c for c in self.columns if not only or c.name in only]
        head = f'TABLE "{self.name}" ({self.rows:,} rows)'
        if self.description:
            head += f" — {self.description}"
        if self.grain:
            head += f"\n  grain: {self.grain}"
        return head + "\n" + "\n".join(c.compact() for c in cols)


@dataclass
class Relationship:
    """A join candidate found by value-overlap, not by a declared foreign key."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    overlap: float              # share of left values present on the right

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityFlag:
    table: str
    column: str
    severity: str               # warning | serious
    kind: str                   # nulls | duplicates | outliers | mixed_type | constant
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Catalog:
    dataset_id: str
    name: str
    tables: list[TableProfile] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    quality: list[QualityFlag] = field(default_factory=list)
    summary: str = ""           # derived one-paragraph description of the dataset

    def table(self, name: str) -> TableProfile | None:
        low = name.lower()
        return next((t for t in self.tables if t.name.lower() == low), None)

    def to_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "name": self.name,
                "summary": self.summary,
                "tables": [t.to_dict() for t in self.tables],
                "relationships": [r.to_dict() for r in self.relationships],
                "quality": [q.to_dict() for q in self.quality]}


# --- the conversation -------------------------------------------------------
@dataclass
class Exchange:
    """One completed round of chat, as the agents see the history.

    Only what a later turn actually needs to resolve a reference: what was
    asked, roughly what came back, and the query that produced it — so "same
    thing but by zone" has something to be the same as."""
    question: str
    answer: str = ""
    sql: str = ""
    columns: list[str] = field(default_factory=list)

    def compact(self, width: int = 240) -> str:
        out = [f"User: {self.question}"]
        if self.answer:
            answer = " ".join(self.answer.split())
            out.append("Assistant: " + answer[:width]
                       + ("…" if len(answer) > width else ""))
        if self.sql:
            sql = " ".join(self.sql.split())
            out.append("  (ran: " + sql[:width]
                       + ("…" if len(sql) > width else "") + ")")
        if self.columns:
            out.append("  (returned: " + ", ".join(self.columns[:8]) + ")")
        return "\n".join(out)


@dataclass
class Route:
    """The router's read of one message: answer it now, or go get the rows.

    `mode="chat"` is a real answer, not a fallback — "what's in this file?" is
    already answered by the derived schema, and running SQL to reply to "thanks"
    is how a chatbot feels like a form."""
    mode: str = "data"          # chat | data
    reply: str = ""             # the answer itself, when mode == "chat"
    question: str = ""          # the message rewritten to stand on its own
    tables: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)   # measured, not guessed

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SQLAttempt:
    n: int
    sql: str
    ok: bool
    error: str = ""
    critique: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    columns: list[str]
    types: list[str]
    rows: list[list]
    row_count: int
    truncated: bool = False
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def schema_line(self) -> str:
        return ", ".join(f"{c} ({t})" for c, t in zip(self.columns, self.types))

    def as_markdown(self, limit: int = 30) -> str:
        head = " | ".join(self.columns)
        sep = " | ".join("---" for _ in self.columns)
        body = "\n".join(" | ".join("" if v is None else str(v) for v in r)
                         for r in self.rows[:limit])
        more = ""
        if self.row_count > limit:
            more = f"\n… {self.row_count - limit:,} more rows"
        return f"{head}\n{sep}\n{body}{more}"


@dataclass
class ChartSpec:
    """The visualization agent's verdict. `should_chart=False` is a real answer —
    a single number belongs in a stat tile, not a one-bar bar chart."""
    should_chart: bool
    reason: str
    form: str = "none"          # stat|bar|hbar|line|area|grouped_bar|stacked_bar|
                                # share|scatter|table|none
    x: str = ""
    y: str = ""
    series: str = ""
    # Wide results put the comparison in COLUMNS, not rows: `zones, may_gtv,
    # jun_gtv` has no series dimension to point `series` at. Naming several
    # numeric columns here melts them into series at render time.
    y_columns: list[str] = field(default_factory=list)
    color_job: str = "sequential"   # sequential|categorical|diverging|none
    emphasis: str = ""              # x-value to highlight, rest recede to gray
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    value_format: str = "number"    # number|compact|currency|percent
    sort: str = "none"              # none|value_desc|value_asc|x_asc
    limit: int = 0                  # 0 = plot every row; set when categories are folded
    notes: list[str] = field(default_factory=list)   # adjustments made by the validator

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Metrics:
    calls: int = 0
    sql_attempts: int = 0
    repairs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_peak: int = 0
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnResult:
    """Everything one chat turn produced. A `chat`-mode turn has an answer and
    nothing else; a `data`-mode turn carries the query, the rows and the chart."""
    question: str
    route: Route | None = None
    sql: str = ""
    attempts: list[SQLAttempt] = field(default_factory=list)
    result: QueryResult | None = None
    chart: ChartSpec | None = None
    answer: str = ""
    suggestions: list[str] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)

    def as_exchange(self) -> Exchange:
        """Fold this turn into what the next turn's router will read."""
        return Exchange(question=self.question, answer=self.answer, sql=self.sql,
                        columns=list(self.result.columns) if self.result else [])
