"""Context budgeting for a small window.

An uploaded workbook can carry forty tables with three hundred columns between
them. That does not fit in a 16k window alongside a system prompt and room to
answer, so every prompt is sized before it is sent:

    input_allowance = max_context - reserved_output - safety_margin

Three fitting strategies, one per shape of thing we have to squeeze in:

  fit_schema()  -- for the router and the suggester, which need *breadth*: they
                   have to see every table to pick the right ones. Renders tables
                   at decreasing detail (full → columns only → name + row count)
                   until they fit.
  focus_schema() -- for the SQL author, which needs *depth* on a few tables: the
                   router's chosen tables in full, everything else dropped.
  fit_rows()    -- for the narrator and the chart agent, which need the
                   *result*: head rows, plus tail rows when the ordering carries
                   meaning, with the middle elided.

Token counts use a conservative chars-per-token heuristic; the safety margin
covers the estimate being wrong. Swap in a real tokenizer at estimate_tokens().
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from models import Catalog, QueryResult, TableProfile

CHARS_PER_TOKEN = 3.8      # conservative for English + SQL identifiers


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " …[truncated]"


def _skeleton(t: TableProfile) -> str:
    """Middle detail level: column names and roles, no stats or samples."""
    cols = ", ".join(f"{c.name} {c.sql_type}" for c in t.columns)
    head = f'TABLE "{t.name}" ({t.rows:,} rows)'
    if t.description:
        head += f" — {t.description}"
    return f"{head}\n  {cols}"


def _stub(t: TableProfile) -> str:
    """Least detail: enough for the planner to know the table exists."""
    return f'TABLE "{t.name}" ({t.rows:,} rows, {len(t.columns)} columns)'


def fit_schema(catalog: Catalog, max_tokens: int,
               prefer: list[str] | None = None) -> tuple[str, int, int]:
    """Render the whole catalog under `max_tokens`, degrading detail rather than
    hiding tables — the planner cannot choose a table it was never shown.

    Returns (text, tables_shown_in_full, total_tables)."""
    tables = list(catalog.tables)
    if not tables:
        return "(no tables)", 0, 0

    preferred = {p.lower() for p in (prefer or [])}
    order = sorted(range(len(tables)),
                   key=lambda i: (tables[i].name.lower() not in preferred, i))

    rendered = {i: tables[i].compact() for i in range(len(tables))}
    full = set(range(len(tables)))

    def total() -> int:
        return estimate_tokens("\n\n".join(
            rendered[i] for i in range(len(tables)))) + _rel_tokens(catalog)

    # Degrade least-preferred tables first: full → skeleton → stub.
    for level in (_skeleton, _stub):
        for i in reversed(order):
            if total() <= max_tokens:
                break
            rendered[i] = level(tables[i])
            full.discard(i)
        if total() <= max_tokens:
            break

    text = "\n\n".join(rendered[i] for i in range(len(tables)))
    rels = _relationship_block(catalog)
    if rels and estimate_tokens(text + rels) <= max_tokens:
        text += rels
    return truncate_to_tokens(text, max_tokens), len(full), len(tables)


def _relationship_block(catalog: Catalog) -> str:
    if not catalog.relationships:
        return ""
    lines = [f"  {r.from_table}.{r.from_column} = {r.to_table}.{r.to_column} "
             f"({r.overlap:.0%} overlap)" for r in catalog.relationships[:20]]
    return "\n\nJOINABLE ON (inferred from value overlap, not declared keys):\n" \
           + "\n".join(lines)


def _rel_tokens(catalog: Catalog) -> int:
    return estimate_tokens(_relationship_block(catalog))


def focus_schema(catalog: Catalog, tables: list[str], max_tokens: int) -> str:
    """Full detail on the planner's chosen tables. If they alone overflow, trim
    the columns the planner did not name."""
    chosen = [t for t in catalog.tables
              if t.name.lower() in {n.lower() for n in tables}]
    if not chosen:
        chosen = catalog.tables[:2]

    text = "\n\n".join(t.compact() for t in chosen) + _relationship_block(catalog)
    if estimate_tokens(text) <= max_tokens:
        return text

    share = max_tokens // max(1, len(chosen))
    parts = []
    for t in chosen:
        block = t.compact()
        if estimate_tokens(block) > share:
            block = _skeleton(t)
        parts.append(truncate_to_tokens(block, share))
    return "\n\n".join(parts)


def fit_rows(result: QueryResult, max_tokens: int,
             head: int = 40, tail: int = 10) -> str:
    """Result rows as a compact table. Ordered results carry meaning at both
    ends (top sellers *and* worst performers), so keep a tail and elide the
    middle rather than truncating flat."""
    if not result.columns:
        return "(no columns)"
    if not result.rows:
        return f"{' | '.join(result.columns)}\n(0 rows)"

    def render(rows: list[list], eliding: int) -> str:
        lines = [" | ".join(result.columns),
                 " | ".join("---" for _ in result.columns)]
        for r in rows[:head]:
            lines.append(" | ".join(_cell(v) for v in r))
        if eliding:
            lines.append(f"… {eliding:,} rows omitted …")
            for r in rows[-tail:]:
                lines.append(" | ".join(_cell(v) for v in r))
        return "\n".join(lines)

    n = len(result.rows)
    while head > 3:
        eliding = max(0, n - head - (tail if n > head + tail else 0))
        shown = result.rows if not eliding else result.rows
        text = render(shown, eliding)
        if estimate_tokens(text) <= max_tokens:
            note = ""
            if result.truncated:
                note = "\n(result was capped — more rows exist)"
            return text + note
        head = int(head * 0.6)
        tail = max(2, int(tail * 0.6))
    return truncate_to_tokens(render(result.rows, 0), max_tokens)


_WS = re.compile(r"\s+")


def _cell(v) -> str:
    if v is None:
        return ""
    s = _WS.sub(" ", str(v))
    return s if len(s) <= 48 else s[:45] + "…"


@dataclass
class Budget:
    max_context: int = 16000
    safety_margin: int = 500

    def input_allowance(self, output_tokens: int) -> int:
        """Tokens left for the prompt once the reply and the margin are reserved."""
        return max(0, self.max_context - output_tokens - self.safety_margin)

    def room_for(self, output_tokens: int, overhead_tokens: int) -> int:
        """Tokens left for the variable payload (schema or rows) after the fixed
        system prompt and question have been accounted for."""
        return max(0, self.input_allowance(output_tokens) - overhead_tokens)
