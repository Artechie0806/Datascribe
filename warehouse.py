"""Upload → DuckDB. One warehouse file per dataset, queried read-only.

Everything the user uploads (csv, tsv, xlsx/xls, sqlite/.db, .duckdb, parquet,
json/jsonl) is converted into tables inside a single DuckDB file that this module
owns. Nothing is pre-loaded and nothing is bundled — a dataset exists only
because someone uploaded it.

Two halves:

  ingest()  — writer. Runs once per upload, converts each source into DuckDB
              tables, and is the ONLY code path that opens the file for writing.
  Warehouse — reader. Opens the same file `read_only=True` with external file
              access disabled, so an LLM-authored query cannot write, attach
              another database, or read a path off this machine. On top of that
              hard boundary sits a statement guard, a row cap, and a wall-clock
              timeout enforced with DuckDB's interrupt.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

MAX_ROWS = 5_000          # hard cap on rows handed back from any single query
QUERY_TIMEOUT = 30.0      # seconds; enforced via con.interrupt()
SAMPLE_ROWS = 5

CSV_EXT = {".csv", ".tsv", ".txt"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}
SQLITE_EXT = {".db", ".sqlite", ".sqlite3", ".db3"}
DUCKDB_EXT = {".duckdb", ".ddb"}
PARQUET_EXT = {".parquet", ".pq"}
JSON_EXT = {".json", ".jsonl", ".ndjson"}
SUPPORTED = (CSV_EXT | EXCEL_EXT | SQLITE_EXT | DUCKDB_EXT | PARQUET_EXT
             | JSON_EXT)

# Anything that writes, reaches outside the database, or loads code. The
# read-only connection already refuses these; the guard exists so the agent gets
# a clear, correctable error instead of a driver exception.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|truncate|attach|detach|copy|"
    r"install|load|export|import|set|reset|call|checkpoint|vacuum|"
    r"read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_xlsx|"
    r"glob|sniff_csv)\b",
    re.IGNORECASE,
)
_LEADING_OK = re.compile(r"^\s*(with|select|explain|describe|summarize|pragma\s+table_info)\b",
                         re.IGNORECASE)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LIT = re.compile(r"'(?:[^']|'')*'")


class IngestError(RuntimeError):
    pass


class QueryError(RuntimeError):
    """A query that was rejected or failed — the message is fed back to the
    repair agent verbatim, so it must stay readable."""


# --- naming -----------------------------------------------------------------
def _slug(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", (name or "").strip()).strip("_").lower()
    if not s:
        s = "table"
    if s[0].isdigit():
        s = "t_" + s
    return s[:56]


def _unique(base: str, taken: set[str]) -> str:
    name, n = base, 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name


def _qi(name: str) -> str:
    """Quote an identifier for interpolation into DDL we generate ourselves."""
    return '"' + name.replace('"', '""') + '"'


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


# --- ingestion (the only writer) --------------------------------------------
@dataclass
class IngestedTable:
    name: str
    rows: int
    source_file: str


def _load_csv(con: duckdb.DuckDBPyConnection, path: Path, table: str) -> None:
    """Typed auto-detect first; retry as all-text so a single dirty column can't
    lose the whole file."""
    src = _sql_str(str(path))
    try:
        con.execute(f"CREATE TABLE {_qi(table)} AS "
                    f"SELECT * FROM read_csv_auto({src}, sample_size=-1, "
                    f"ignore_errors=true)")
    except duckdb.Error:
        con.execute(f"CREATE TABLE {_qi(table)} AS "
                    f"SELECT * FROM read_csv({src}, all_varchar=true, "
                    f"ignore_errors=true, header=true)")


def _load_excel(con: duckdb.DuckDBPyConnection, path: Path, stem: str,
                taken: set[str]) -> list[str]:
    """One table per sheet. pandas + openpyxl rather than DuckDB's excel
    extension: it needs no network install and gives us per-sheet control."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise IngestError("pandas is required to read Excel files") from e

    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        raise IngestError(f"could not read workbook: {e}") from e

    made: list[str] = []
    for sheet, df in sheets.items():
        if df.empty and not len(df.columns):
            continue
        df = _clean_frame(df)
        base = _slug(sheet) if len(sheets) > 1 else _slug(stem)
        if len(sheets) > 1 and base.startswith(("sheet", "table")):
            base = _slug(f"{stem}_{sheet}")
        name = _unique(base, taken)
        con.register("_incoming", df)
        con.execute(f"CREATE TABLE {_qi(name)} AS SELECT * FROM _incoming")
        con.unregister("_incoming")
        made.append(name)
    if not made:
        raise IngestError("workbook contained no readable sheets")
    return made


def _clean_frame(df):
    """Excel sheets arrive with blank spacer columns and duplicate headers;
    normalise them so the resulting SQL columns are addressable."""
    import pandas as pd

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    cols, taken = [], set()
    for i, c in enumerate(df.columns):
        base = _slug(str(c)) if not (c is None or (isinstance(c, float) and pd.isna(c))) else ""
        if not base or base.startswith("unnamed"):
            base = f"column_{i + 1}"
        cols.append(_unique(base, taken))
    df.columns = cols
    return df


def _load_sqlite(con: duckdb.DuckDBPyConnection, path: Path,
                 taken: set[str]) -> list[str]:
    """Prefer DuckDB's sqlite_scanner (types survive); fall back to stdlib
    sqlite3 + pandas when the extension can't be installed offline."""
    try:
        con.execute("INSTALL sqlite")
        con.execute("LOAD sqlite")
        con.execute(f"ATTACH {_sql_str(str(path))} AS _src (TYPE sqlite, READ_ONLY)")
    except duckdb.Error:
        return _load_sqlite_stdlib(con, path, taken)

    made: list[str] = []
    try:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = '_src' AND table_schema = 'main'").fetchall()
        for (src_table,) in rows:
            if src_table.startswith("sqlite_"):
                continue
            name = _unique(_slug(src_table), taken)
            con.execute(f"CREATE TABLE {_qi(name)} AS "
                        f"SELECT * FROM _src.main.{_qi(src_table)}")
            made.append(name)
    finally:
        try:
            con.execute("DETACH _src")
        except duckdb.Error:
            pass
    if not made:
        raise IngestError("database contained no user tables")
    return made


def _load_sqlite_stdlib(con: duckdb.DuckDBPyConnection, path: Path,
                        taken: set[str]) -> list[str]:
    import pandas as pd

    made: list[str] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as lite:
        names = [r[0] for r in lite.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'").fetchall()]
        for src_table in names:
            df = pd.read_sql_query(f'SELECT * FROM "{src_table}"', lite)
            name = _unique(_slug(src_table), taken)
            con.register("_incoming", _clean_frame(df))
            con.execute(f"CREATE TABLE {_qi(name)} AS SELECT * FROM _incoming")
            con.unregister("_incoming")
            made.append(name)
    if not made:
        raise IngestError("database contained no user tables")
    return made


def _load_duckdb(con: duckdb.DuckDBPyConnection, path: Path,
                 taken: set[str]) -> list[str]:
    """Copy an uploaded DuckDB file's tables into our warehouse rather than
    querying it in place — we keep one file per dataset and one writer."""
    con.execute(f"ATTACH {_sql_str(str(path))} AS _src (READ_ONLY)")
    made: list[str] = []
    try:
        rows = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_catalog = '_src'").fetchall()
        for schema, src_table in rows:
            name = _unique(_slug(src_table), taken)
            con.execute(f"CREATE TABLE {_qi(name)} AS "
                        f"SELECT * FROM _src.{_qi(schema)}.{_qi(src_table)}")
            made.append(name)
    finally:
        try:
            con.execute("DETACH _src")
        except duckdb.Error:
            pass
    if not made:
        raise IngestError("database contained no tables")
    return made


def ingest(sources: list[Path], warehouse_path: Path) -> list[IngestedTable]:
    """Convert every uploaded file into tables in a fresh DuckDB warehouse.

    `sources` are already-saved temp paths; `warehouse_path` is created (and
    replaced if it exists). Returns one entry per table created."""
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    if warehouse_path.exists():
        warehouse_path.unlink()

    con = duckdb.connect(str(warehouse_path))
    taken: set[str] = set()
    made: list[tuple[str, str]] = []   # (table, source filename)
    errors: list[str] = []
    try:
        for path in sources:
            ext = path.suffix.lower()
            stem = path.stem
            try:
                if ext in CSV_EXT:
                    name = _unique(_slug(stem), taken)
                    _load_csv(con, path, name)
                    made.append((name, path.name))
                elif ext in EXCEL_EXT:
                    for name in _load_excel(con, path, stem, taken):
                        made.append((name, path.name))
                elif ext in SQLITE_EXT:
                    for name in _load_sqlite(con, path, taken):
                        made.append((name, path.name))
                elif ext in DUCKDB_EXT:
                    for name in _load_duckdb(con, path, taken):
                        made.append((name, path.name))
                elif ext in PARQUET_EXT:
                    name = _unique(_slug(stem), taken)
                    con.execute(f"CREATE TABLE {_qi(name)} AS SELECT * FROM "
                                f"read_parquet({_sql_str(str(path))})")
                    made.append((name, path.name))
                elif ext in JSON_EXT:
                    name = _unique(_slug(stem), taken)
                    con.execute(f"CREATE TABLE {_qi(name)} AS SELECT * FROM "
                                f"read_json_auto({_sql_str(str(path))})")
                    made.append((name, path.name))
                else:
                    errors.append(f"{path.name}: unsupported file type '{ext}'")
            except (duckdb.Error, IngestError) as e:
                errors.append(f"{path.name}: {e}")

        if not made:
            raise IngestError("; ".join(errors) or "no tables could be created")

        out: list[IngestedTable] = []
        for name, source in made:
            rows = con.execute(f"SELECT count(*) FROM {_qi(name)}").fetchone()[0]
            out.append(IngestedTable(name=name, rows=int(rows), source_file=source))
        return out
    finally:
        con.close()


# --- query guard ------------------------------------------------------------
def _strip_noise(sql: str) -> str:
    """Comments and string literals removed, so the keyword guard can't be
    fooled by a comment or by the word 'update' inside a filter value."""
    s = _BLOCK_COMMENT.sub(" ", sql)
    s = _LINE_COMMENT.sub(" ", s)
    return _STRING_LIT.sub("''", s)


def guard(sql: str) -> str:
    """Return the single read-only statement, or raise QueryError explaining why
    it was rejected. The message goes straight to the repair agent."""
    if not sql or not sql.strip():
        raise QueryError("empty query")
    bare = _strip_noise(sql).strip().rstrip(";").strip()
    if not bare:
        raise QueryError("query contained no statement")
    if ";" in bare:
        raise QueryError("only one statement is allowed per query")
    if not _LEADING_OK.match(bare):
        raise QueryError("only SELECT / WITH queries are allowed")
    hit = _FORBIDDEN.search(bare)
    if hit:
        raise QueryError(
            f"'{hit.group(0)}' is not allowed — this connection is read-only and "
            f"can only query the tables already in the warehouse")
    return sql.strip().rstrip(";").strip()


# --- reader -----------------------------------------------------------------
class Warehouse:
    """Read-only handle on one dataset's DuckDB file."""

    def __init__(self, path: Path, timeout: float = QUERY_TIMEOUT,
                 max_rows: int = MAX_ROWS):
        self.path = Path(path)
        self.timeout = timeout
        self.max_rows = max_rows
        if not self.path.exists():
            raise QueryError(f"warehouse not found: {self.path.name}")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        # enable_external_access=false is the real boundary: no ATTACH, no
        # httpfs, no reading a file off this machine from inside a query.
        return duckdb.connect(str(self.path), read_only=True, config={
            "enable_external_access": "false",
        })

    def _run(self, con: duckdb.DuckDBPyConnection, sql: str):
        """Execute with a wall-clock timeout. DuckDB has no query_timeout
        setting, so a watchdog thread interrupts the connection instead."""
        timer = threading.Timer(self.timeout, con.interrupt)
        timer.daemon = True
        timer.start()
        try:
            return con.execute(sql)
        except duckdb.InterruptException:
            raise QueryError(
                f"query exceeded the {self.timeout:.0f}s limit — narrow it with a "
                f"filter, an aggregate, or a LIMIT") from None
        finally:
            timer.cancel()

    def query(self, sql: str):
        """Guarded, capped, timed read. Returns a models.QueryResult."""
        from models import QueryResult

        clean = guard(sql)
        started = time.perf_counter()
        con = self._connect()
        try:
            cur = self._run(con, clean)
            columns = [d[0] for d in cur.description]
            types = [str(d[1]) for d in cur.description]
            rows = cur.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            rows = rows[: self.max_rows]
        except duckdb.Error as e:
            raise QueryError(str(e).strip()) from None
        finally:
            con.close()

        return QueryResult(
            columns=columns, types=types,
            rows=[[_jsonable(v) for v in r] for r in rows],
            row_count=len(rows), truncated=truncated,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def explain(self, sql: str) -> str:
        """Plan-only dry run: catches unknown columns and bad joins without
        paying for execution."""
        clean = guard(sql)
        con = self._connect()
        try:
            cur = self._run(con, f"EXPLAIN {clean}")
            return "\n".join(str(r[-1]) for r in cur.fetchall())
        except duckdb.Error as e:
            raise QueryError(str(e).strip()) from None
        finally:
            con.close()

    # --- profiling support (trusted, internally-generated SQL) --------------
    def introspect(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run SQL this module generated itself (profiling, schema reads). Not
        for model-authored queries — those go through query()."""
        con = self._connect()
        try:
            return self._run_params(con, sql, params).fetchall()
        except duckdb.Error as e:
            raise QueryError(str(e).strip()) from None
        finally:
            con.close()

    def _run_params(self, con, sql: str, params: tuple):
        timer = threading.Timer(self.timeout, con.interrupt)
        timer.daemon = True
        timer.start()
        try:
            return con.execute(sql, params) if params else con.execute(sql)
        except duckdb.InterruptException:
            raise QueryError("profiling query timed out") from None
        finally:
            timer.cancel()

    def tables(self) -> list[str]:
        rows = self.introspect(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name")
        return [r[0] for r in rows]

    def columns(self, table: str) -> list[tuple[str, str]]:
        rows = self.introspect(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? "
            "ORDER BY ordinal_position", (table,))
        return [(r[0], r[1]) for r in rows]


def _jsonable(v):
    """DuckDB hands back Decimal/date/UUID/bytes — make them JSON-safe without
    losing precision the UI would want to show."""
    import datetime
    import decimal
    import uuid

    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, decimal.Decimal):
        f = float(v)
        return int(f) if f.is_integer() else f
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, datetime.timedelta):
        return str(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(v))} bytes>"
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def dataset_dir(root: Path, dataset_id: str) -> Path:
    return root / dataset_id


def destroy(root: Path, dataset_id: str) -> None:
    """Delete a dataset's warehouse and its uploaded originals."""
    target = dataset_dir(root, dataset_id)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
