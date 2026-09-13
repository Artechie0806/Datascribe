"""The catalog builder — this is what replaces a hand-written data dictionary.

Nothing about any dataset is known ahead of time, so the meaning of a table has
to be *derived* at upload time. Two passes:

  1. Statistical (deterministic, no LLM). Types, null rates, cardinality,
     uniqueness, ranges, real sample values, duplicate rows, outliers, and
     join candidates found by measuring value overlap between columns — not by
     trusting a declared foreign key, because a CSV has none.

  2. Semantic (one LLM call per dataset). The statistics are handed to the model,
     which writes the table/column descriptions, each table's grain, and a
     one-paragraph dataset summary. Those descriptions are what later agents read
     when they pick tables and write SQL.

Pass 1 is the ground truth; pass 2 only adds prose. If the LLM is unreachable the
catalog is still complete and the pipeline still works — it just reads drier.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from llm import QwenClient
from models import Catalog, ColumnProfile, QualityFlag, Relationship, TableProfile
from warehouse import Warehouse

SAMPLE_ROWS = 400          # rows drawn per table to source example values
MAX_SAMPLES = 5            # example values shown per column
EXACT_DISTINCT_LIMIT = 300_000
DUP_CHECK_LIMIT = 500_000
MAX_JOIN_PROBES = 40       # cap the O(n²) column-pair scan on wide datasets
HIGH_NULL_PCT = 40.0
PROFILE_WORKERS = 4

_NUMERIC = re.compile(
    r"^(tinyint|smallint|integer|bigint|hugeint|utinyint|usmallint|uinteger|"
    r"ubigint|float|real|double|decimal|numeric)", re.IGNORECASE)
_FRACTIONAL = re.compile(r"^(float|real|double|decimal|numeric)", re.IGNORECASE)
_TEMPORAL = re.compile(r"^(date|timestamp|time|interval)", re.IGNORECASE)
_TEXTUAL = re.compile(r"^(varchar|char|text|string|uuid|blob)", re.IGNORECASE)
# Key-ish names come in two conventions and both have to be caught. The snake
# form is case-insensitive (`customer_id`, `ORDER_KEY`, `id`); the camel form has
# to stay case-SENSITIVE, because folding case there would make "paid" and
# "void" end in an id and turn every such column into a key.
_ID_SNAKE = re.compile(r"(^|_)(id|key|code|uuid|guid|no|number)$", re.IGNORECASE)
_ID_CAMEL = re.compile(r"[a-z0-9](Id|Key|Code|Uuid|Guid|No|Number)$")


def _looks_like_id(col: str) -> bool:
    """`customer_id` and `CustomerId` are the same foreign key wearing different
    house styles — and Chinook, Northwind and anything born in SQL Server wear
    the second one."""
    return bool(_ID_SNAKE.search(col) or _ID_CAMEL.search(col))
_NUMERIC_TEXT = re.compile(r"^-?[\d,]+(\.\d+)?$")


def is_numeric(t: str) -> bool:
    return bool(_NUMERIC.match(t or ""))


def is_temporal(t: str) -> bool:
    return bool(_TEMPORAL.match(t or ""))


def is_textual(t: str) -> bool:
    return bool(_TEXTUAL.match(t or ""))


def _qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _role(col: str, sql_type: str, distinct: int, rows: int, unique: bool,
          longest_sample: int = 0) -> str:
    """Classify a column by what an analyst would *do* with it.

    The traps this avoids, in order of how often they bite:
    - `customer_id` is a foreign key even though it repeats and looks numeric.
      Summing it is meaningless, so the name wins over the statistics.
    - A unique DOUBLE (`price` in a 20-row product table) is a measure that
      happens not to repeat — uniqueness only implies a key for whole numbers.
    - A numeric column with a handful of distinct values (`signup_year`,
      `status = 1|2|3`) is a category encoded as a number.
    """
    if is_temporal(sql_type):
        return "temporal"
    name_is_id = _looks_like_id(col)

    if is_numeric(sql_type):
        if name_is_id and distinct > 1:
            return "identifier"
        if unique and rows > 1 and not _FRACTIONAL.match(sql_type):
            return "identifier"
        if distinct <= 12 and rows >= 20:
            return "dimension"
        return "measure"

    if sql_type.upper().startswith("BOOLEAN"):
        return "dimension"
    if is_textual(sql_type):
        # A unique short string is a key or a label; a unique long one is prose.
        if unique and rows > 1:
            return "identifier" if longest_sample <= 40 else "text"
        if name_is_id and distinct > max(20, rows * 0.4):
            return "identifier"
        if distinct <= max(50, rows * 0.05):
            return "dimension"
        return "text"
    return "dimension"


def _fmt(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "…"


# --- pass 1: statistics -----------------------------------------------------
def _profile_table(wh: Warehouse, table: str, source_file: str) -> tuple[TableProfile, list[QualityFlag]]:
    cols = wh.columns(table)
    rows = int(wh.introspect(f"SELECT count(*) FROM {_qi(table)}")[0][0])
    flags: list[QualityFlag] = []

    if not rows or not cols:
        profile = TableProfile(name=table, rows=rows, source_file=source_file,
                               columns=[ColumnProfile(name=c, sql_type=t)
                                        for c, t in cols])
        return profile, flags

    exact = rows <= EXACT_DISTINCT_LIMIT
    parts: list[str] = []
    for name, sql_type in cols:
        q = _qi(name)
        parts.append(f"count({q})")
        parts.append(f"count(DISTINCT {q})" if exact
                     else f"approx_count_distinct({q})")
        if is_numeric(sql_type) or is_temporal(sql_type):
            parts.append(f"min({q})")
            parts.append(f"max({q})")
        else:
            parts.append("NULL")
            parts.append("NULL")
        parts.append(f"avg({q})" if is_numeric(sql_type) else "NULL")
    stats = wh.introspect(f"SELECT {', '.join(parts)} FROM {_qi(table)}")[0]

    samples = _sample_values(wh, table, [c for c, _ in cols])

    columns: list[ColumnProfile] = []
    for i, (name, sql_type) in enumerate(cols):
        non_null, distinct, lo, hi, mean = stats[i * 5: i * 5 + 5]
        non_null = int(non_null or 0)
        distinct = int(distinct or 0)
        nulls = rows - non_null
        unique = distinct == non_null and non_null == rows and rows > 1
        seen = samples.get(name, [])
        cp = ColumnProfile(
            name=name, sql_type=str(sql_type),
            role=_role(name, str(sql_type), distinct, rows, unique,
                       max((len(s) for s in seen), default=0)),
            nulls=nulls, null_pct=round(100.0 * nulls / rows, 2),
            distinct=distinct, is_unique=unique,
            min=_fmt(lo), max=_fmt(hi),
            mean=round(float(mean), 4) if mean is not None else None,
            samples=seen,
        )
        columns.append(cp)
        flags.extend(_column_flags(table, cp, rows))

    flags.extend(_outlier_flags(wh, table, columns))
    flags.extend(_duplicate_flag(wh, table, rows, [c for c, _ in cols]))
    return TableProfile(name=table, rows=rows, columns=columns,
                        source_file=source_file), flags


def _sample_values(wh: Warehouse, table: str,
                   names: list[str]) -> dict[str, list[str]]:
    """Real values pulled from one reservoir sample of the table — one query for
    every column, instead of a per-column scan."""
    try:
        picked = wh.introspect(
            f"SELECT * FROM {_qi(table)} USING SAMPLE {SAMPLE_ROWS} ROWS")
    except Exception:
        try:
            picked = wh.introspect(f"SELECT * FROM {_qi(table)} LIMIT {SAMPLE_ROWS}")
        except Exception:
            return {}

    out: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        seen: list[str] = []
        for row in picked:
            if i >= len(row):
                break
            v = row[i]
            if v is None:
                continue
            s = _fmt(v)
            if s not in seen:
                seen.append(s)
            if len(seen) >= MAX_SAMPLES:
                break
        out[name] = seen
    return out


def _column_flags(table: str, cp: ColumnProfile, rows: int) -> list[QualityFlag]:
    flags: list[QualityFlag] = []
    if cp.null_pct >= HIGH_NULL_PCT:
        flags.append(QualityFlag(
            table, cp.name, "serious" if cp.null_pct >= 80 else "warning",
            "nulls", f"{cp.null_pct:.0f}% of values are missing "
                     f"({cp.nulls:,} of {rows:,} rows)"))
    if cp.distinct <= 1 and rows > 1 and cp.nulls < rows:
        flags.append(QualityFlag(
            table, cp.name, "warning", "constant",
            "every row holds the same value — it cannot separate anything"))
    if is_textual(cp.sql_type) and cp.samples:
        numeric_like = sum(1 for s in cp.samples if _NUMERIC_TEXT.match(s))
        if numeric_like == len(cp.samples) and cp.distinct > 5:
            flags.append(QualityFlag(
                table, cp.name, "warning", "mixed_type",
                "stored as text but the values look numeric — comparisons and "
                "sorts will be alphabetical unless it is cast"))
    return flags


def _outlier_flags(wh: Warehouse, table: str,
                   columns: list[ColumnProfile]) -> list[QualityFlag]:
    """Tukey fences on measures. Outliers are not errors, but an analyst should
    be told before an average gets quoted as a headline."""
    measures = [c for c in columns if c.role == "measure" and c.distinct > 10]
    if not measures:
        return []
    parts = []
    for c in measures:
        q = _qi(c.name)
        parts.append(f"quantile_cont({q}, 0.25)")
        parts.append(f"quantile_cont({q}, 0.75)")
    try:
        qs = wh.introspect(f"SELECT {', '.join(parts)} FROM {_qi(table)}")[0]
    except Exception:
        return []

    checks, kept = [], []
    for i, c in enumerate(measures):
        p25, p75 = qs[i * 2], qs[i * 2 + 1]
        if p25 is None or p75 is None:
            continue
        iqr = float(p75) - float(p25)
        if iqr <= 0:
            continue
        lo, hi = float(p25) - 1.5 * iqr, float(p75) + 1.5 * iqr
        checks.append(f"count(*) FILTER (WHERE {_qi(c.name)} < {lo} "
                      f"OR {_qi(c.name)} > {hi})")
        kept.append((c, lo, hi))
    if not checks:
        return []
    try:
        counts = wh.introspect(f"SELECT {', '.join(checks)} FROM {_qi(table)}")[0]
    except Exception:
        return []

    flags = []
    for (c, lo, hi), n in zip(kept, counts):
        n = int(n or 0)
        if n:
            flags.append(QualityFlag(
                table, c.name, "warning", "outliers",
                f"{n:,} values fall outside the Tukey fence "
                f"[{lo:,.4g}, {hi:,.4g}] — a mean over this column is skewed"))
    return flags


def _duplicate_flag(wh: Warehouse, table: str, rows: int,
                    names: list[str]) -> list[QualityFlag]:
    if rows > DUP_CHECK_LIMIT or rows < 2 or not names:
        return []
    try:
        distinct = int(wh.introspect(
            f"SELECT count(*) FROM (SELECT DISTINCT * FROM {_qi(table)})")[0][0])
    except Exception:
        return []
    dupes = rows - distinct
    if dupes <= 0:
        return []
    return [QualityFlag(
        table, "*", "serious" if dupes > rows * 0.05 else "warning",
        "duplicates",
        f"{dupes:,} fully duplicated rows ({100.0 * dupes / rows:.1f}%) — counts "
        f"and sums will be inflated unless they are de-duplicated")]


# --- pass 1b: join discovery by value overlap -------------------------------
def _find_relationships(wh: Warehouse,
                        tables: list[TableProfile]) -> list[Relationship]:
    """A CSV has no foreign keys, so infer them: for each unique-valued column,
    look for a column elsewhere whose values largely land inside it."""
    if len(tables) < 2:
        return []

    keys = [(t, c) for t in tables for c in t.columns
            if c.is_unique and c.role in ("identifier", "dimension")]
    candidates: list[tuple] = []
    for kt, kc in keys:
        for t in tables:
            if t.name == kt.name:
                continue
            for c in t.columns:
                if c.role not in ("identifier", "dimension") or c.distinct < 2:
                    continue
                if is_numeric(kc.sql_type) != is_numeric(c.sql_type):
                    continue
                name_match = (c.name.lower() == kc.name.lower()
                              or kc.name.lower() in c.name.lower()
                              or c.name.lower() in kc.name.lower())
                # Two unrelated integer ranges overlap by coincidence all the
                # time — order_id 1..500 "contains" customer_id 1..60 and means
                # nothing by it. For numbers, demand that the names agree too;
                # for strings and uuids, value overlap is evidence on its own.
                if not name_match and is_numeric(c.sql_type):
                    continue
                # Unique on both sides with different names is a coincidence,
                # not a 1:1 relationship.
                if not name_match and c.is_unique:
                    continue
                candidates.append((0 if name_match else 1, t, c, kt, kc))

    candidates.sort(key=lambda x: x[0])   # probe name matches first
    found: list[Relationship] = []
    seen_pairs: set[frozenset] = set()
    for _, t, c, kt, kc in candidates[:MAX_JOIN_PROBES]:
        pair = frozenset({(t.name, c.name), (kt.name, kc.name)})
        if pair in seen_pairs:
            continue
        sql = (f'SELECT count(*), count(*) FILTER (WHERE EXISTS '
               f'(SELECT 1 FROM {_qi(kt.name)} k WHERE k.{_qi(kc.name)} = l.v)) '
               f'FROM (SELECT DISTINCT {_qi(c.name)} AS v FROM {_qi(t.name)} '
               f'WHERE {_qi(c.name)} IS NOT NULL) l')
        try:
            total, hit = wh.introspect(sql)[0]
        except Exception:
            continue
        total, hit = int(total or 0), int(hit or 0)
        if total and hit / total >= 0.8:
            seen_pairs.add(pair)
            found.append(Relationship(t.name, c.name, kt.name, kc.name,
                                      round(hit / total, 3)))
    return found


# --- pass 2: semantic descriptions (the derived dictionary) -----------------
_SEMANTIC_SYSTEM = (
    "You are a data engineer documenting an unfamiliar dataset. You are given "
    "measured statistics for each table and column — types, null rates, "
    "cardinality, ranges and real sample values. Describe what the data IS, "
    "using only what those statistics show. Never invent business context that "
    "the column names and values do not support; if a column's meaning is "
    "genuinely unclear, say so rather than guessing."
)

_TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "grain": {"type": "string"},
        "columns": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "description": {"type": "string"}},
            "required": ["name", "description"]}},
    },
    "required": ["description", "grain", "columns"],
}

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

# One local server means one model instance: four concurrent describe calls do
# not run four times faster, they queue — and on a small GPU they compete for
# the same KV cache. Tune with LLM_DESCRIBE_WORKERS.
DESCRIBE_WORKERS = max(1, int(os.getenv("LLM_DESCRIBE_WORKERS", "2")))
MAX_DESCRIBED_COLUMNS = 60


def _table_budget(table: TableProfile) -> int:
    """Room for the reply. Every column costs a name and a sentence, so the
    ceiling has to follow the table's width rather than sit at a constant."""
    return min(2600, 320 + 58 * min(len(table.columns), MAX_DESCRIBED_COLUMNS))


def _describe_table(table: TableProfile, catalog: Catalog,
                    client: QwenClient) -> None:
    joins = [f"  {r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column} "
             f"({r.overlap:.0%} of values match)" for r in catalog.relationships
             if table.name in (r.from_table, r.to_table)]
    user = (
        f"Dataset: {catalog.name}\n\n{table.compact()}\n"
        + ("\nMeasured join candidates touching this table (value overlap):\n"
           + "\n".join(joins) + "\n" if joins else "")
        + "\nWrite a one-sentence description of this table, its grain (what "
          "exactly one row represents), and a short description for every "
          "column listed above. Keep every description under 20 words."
    )
    data, _ = client.structured(_SEMANTIC_SYSTEM, user, _TABLE_SCHEMA,
                                _table_budget(table))
    table.description = (data.get("description") or "").strip()
    table.grain = (data.get("grain") or "").strip()
    by_name = {c.name.lower(): c for c in table.columns}
    for cd in data.get("columns", []):
        col = by_name.get(str(cd.get("name", "")).lower())
        if col:
            col.description = (cd.get("description") or "").strip()


def _summarise(catalog: Catalog, client: QwenClient) -> None:
    lines = [f"  {t.name} ({t.rows:,} rows): "
             + (t.description or ", ".join(c.name for c in t.columns[:8]))
             for t in catalog.tables]
    user = (f"Dataset: {catalog.name}\n\nTables:\n" + "\n".join(lines)
            + "\n\nWrite a 2-3 sentence summary of this dataset as a whole, "
              "including the kinds of question it can answer.")
    data, _ = client.structured(_SEMANTIC_SYSTEM, user, _SUMMARY_SCHEMA, 400)
    catalog.summary = (data.get("summary") or "").strip()


def _describe(catalog: Catalog, client: QwenClient) -> None:
    """Fill in descriptions/grain/summary in place, one call per table.

    One call for the whole catalog cannot survive a real workbook: the reply has
    to carry a line for every column of every table, so an eleven-table upload
    needs several thousand output tokens, comes back truncated mid-JSON, fails to
    parse — and the whole semantic layer is silently lost, leaving a schema panel
    with no grain and no descriptions anywhere.

    Per-table calls stay inside a budget that follows the table's width, run
    concurrently, and fail independently: one awkward table costs its own
    descriptions and nothing else. Still best-effort — the statistical catalog
    from pass 1 stands on its own if the model is unreachable."""
    if not catalog.tables:
        return
    workers = min(DESCRIBE_WORKERS, len(catalog.tables))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_describe_table, t, catalog, client)
                   for t in catalog.tables]
        for fut in futures:
            try:
                fut.result()
            except Exception:
                continue
    try:
        _summarise(catalog, client)
    except Exception:
        pass


# --- entry point ------------------------------------------------------------
def build_catalog(wh: Warehouse, dataset_id: str, name: str,
                  sources: dict[str, str], client: QwenClient | None = None,
                  describe: bool = True) -> Catalog:
    """Profile every table in the warehouse and return the derived catalog.

    `sources` maps table name -> originating filename."""
    names = wh.tables()
    tables: list[TableProfile] = []
    quality: list[QualityFlag] = []

    with ThreadPoolExecutor(max_workers=min(PROFILE_WORKERS, max(1, len(names)))) as pool:
        futures = [pool.submit(_profile_table, wh, t, sources.get(t, ""))
                   for t in names]
        for fut in futures:
            table, flags = fut.result()
            tables.append(table)
            quality.extend(flags)

    tables.sort(key=lambda t: names.index(t.name))
    catalog = Catalog(dataset_id=dataset_id, name=name, tables=tables,
                      relationships=_find_relationships(wh, tables),
                      quality=quality)
    if describe:
        _describe(catalog, client or QwenClient())
    return catalog


def save(catalog: Catalog, path) -> None:
    path.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")


def load(path) -> Catalog:
    d = json.loads(path.read_text(encoding="utf-8"))
    return Catalog(
        dataset_id=d["dataset_id"], name=d["name"], summary=d.get("summary", ""),
        tables=[TableProfile(
            name=t["name"], rows=t["rows"], source_file=t.get("source_file", ""),
            description=t.get("description", ""), grain=t.get("grain", ""),
            columns=[ColumnProfile(**c) for c in t["columns"]])
            for t in d["tables"]],
        relationships=[Relationship(**r) for r in d.get("relationships", [])],
        quality=[QualityFlag(**q) for q in d.get("quality", [])],
    )
