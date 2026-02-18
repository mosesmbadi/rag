"""
Connect to configured PostgreSQL databases, chunk all tables, and index
them into OpenSearch — one index per database.

Usage:
    python chunk_db.py

Configure databases via DB_CONFIGS in .env:

DB_CONFIGS=[
  {"name":"eqa-monthly","host":"localhost","port":5432,"user":"huqas","password":"huqas","dbname":"eqa-monthly"},
  {"name":"eqa-quarterly","host":"localhost","port":5432,"user":"huqas","password":"huqas","dbname":"eqa-quarterly"}
]

Optional env vars:
    MAX_ROWS_PER_TABLE=50000   # skip tables larger than this (default: 50000)
    LOOKUP_MAX_ROWS=5000       # max rows loaded for FK reference tables (default: 5000)
    STREAM_BATCH=2000          # rows fetched per database cursor batch (default: 2000)

Rule of thumb: always use --fresh after an interrupted run to avoid stale orphan chunks.    
"""
import sys
import os
import re
import json
import psycopg2
import psycopg2.extras
from chunker import (
    build_lookup_tables_from_rows, build_schema_document_from_rows, schema_to_chunks,
    process_rows, ensure_index, bulk_index,
)
from dotenv import load_dotenv

load_dotenv()

MAX_ROWS_PER_TABLE = int(os.getenv('MAX_ROWS_PER_TABLE', '50000'))
LOOKUP_MAX_ROWS    = int(os.getenv('LOOKUP_MAX_ROWS', '5000'))
STREAM_BATCH       = int(os.getenv('STREAM_BATCH', '2000'))


# ── Config ────────────────────────────────────────────────────────────────────

def get_db_configs() -> list[dict]:
    raw = os.getenv('DB_CONFIGS', '[]')
    try:
        configs = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error parsing DB_CONFIGS: {e}")
        sys.exit(1)
    if not configs:
        print("No databases configured. Set DB_CONFIGS in .env")
        sys.exit(1)
    return configs


def connect(config: dict):
    return psycopg2.connect(
        host=config['host'],
        port=int(config.get('port', 5432)),
        user=config['user'],
        password=config['password'],
        dbname=config['dbname'],
        connect_timeout=10,
    )


def db_name_to_index(db_name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', db_name.lower()).strip('_') + '_index'


# ── Schema introspection ──────────────────────────────────────────────────────

def get_table_meta(conn) -> dict[str, dict]:
    """
    Return {table_name: {'columns': [...], 'row_count': int}} for all public tables.
    Uses a fast estimate for row counts (avoids full table scan).
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                t.table_name,
                array_agg(c.column_name ORDER BY c.ordinal_position) AS columns
            FROM information_schema.tables t
            JOIN information_schema.columns c
              ON c.table_name = t.table_name AND c.table_schema = t.table_schema
            WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
            GROUP BY t.table_name
            ORDER BY t.table_name
        """)
        col_rows = cur.fetchall()

        cur.execute("""
            SELECT relname AS table_name,
                   reltuples::bigint AS row_estimate
            FROM pg_class
            WHERE relkind = 'r'
              AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        """)
        counts = {r['table_name']: max(0, r['row_estimate']) for r in cur.fetchall()}

    return {
        r['table_name']: {
            'columns': list(r['columns']),
            'row_count': counts.get(r['table_name'], 0)
        }
        for r in col_rows
    }


# ── FK resolution ─────────────────────────────────────────────────────────────

def fetch_lookup_tables(conn, table_meta: dict) -> dict[str, list[dict]]:
    """
    Fetch only small reference tables (those with both 'id' and 'name' columns).
    Uses an exact COUNT for tables where pg_class estimate is 0 or unreliable.
    """
    lookup_tables = {}
    for tbl, meta in table_meta.items():
        cols = meta['columns']
        if 'id' not in cols or 'name' not in cols:
            continue

        # pg_class estimates can be 0 for unvacuumed tables — do an exact count
        # only for tables that look like small reference tables (estimate ≤ LOOKUP_MAX_ROWS or 0)
        estimated = meta['row_count']
        if estimated > LOOKUP_MAX_ROWS:
            continue  # definitely too large

        try:
            with conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')  # noqa: S608
                exact_count = cur.fetchone()[0]
            if exact_count > LOOKUP_MAX_ROWS:
                continue
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f'SELECT * FROM "{tbl}" LIMIT {LOOKUP_MAX_ROWS}')  # noqa: S608
                lookup_tables[tbl] = [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"    Warning: could not fetch lookup table {tbl}: {e}")

    return lookup_tables


# ── Streaming chunker ─────────────────────────────────────────────────────────

def stream_and_index_table(conn, table_name: str, lookups: dict,
                           index_name: str, start_id: int,
                           total_rows: int = 0) -> int:
    """
    Stream rows from table_name in batches, chunk each batch, embed and index.
    Prints progress every batch. Returns total chunks indexed.
    """
    total_indexed = 0
    rows_seen = 0
    doc_id = start_id

    with conn.cursor(name=f'stream_{table_name}',
                     cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f'SELECT * FROM "{table_name}" LIMIT {MAX_ROWS_PER_TABLE}')  # noqa: S608
        while True:
            batch = cur.fetchmany(STREAM_BATCH)
            if not batch:
                break
            rows = [dict(r) for r in batch]
            rows_seen += len(rows)
            chunks = process_rows(table_name, rows, lookups=lookups)
            if chunks:
                indexed = bulk_index(chunks, index_name, start_id=doc_id)
                doc_id += indexed
                total_indexed += indexed

            progress = f"{rows_seen:,}"
            if total_rows:
                pct = min(100, rows_seen * 100 // total_rows)
                progress += f"/{total_rows:,} ({pct}%)"
            print(f"    {progress} rows processed, {total_indexed} chunk(s) indexed...",
                  end='\r', flush=True)

    print(f"    {rows_seen:,} rows → {total_indexed} chunk(s)               ")
    return total_indexed


# ── Per-database pipeline ─────────────────────────────────────────────────────

def process_database(config: dict, fresh: bool = False):
    db_name = config['name']
    index_name = db_name_to_index(db_name)

    print(f"\n{'='*60}")
    print(f"Database : {db_name}  ({config['host']}:{config.get('port', 5432)})")
    print(f"Index    : {index_name}")

    try:
        conn = connect(config)
    except Exception as e:
        print(f"  Connection failed: {e}")
        return

    # 1. Get schema metadata (fast — queries information_schema + pg_class)
    print("  Reading table metadata...")
    table_meta = get_table_meta(conn)
    print(f"  {len(table_meta)} table(s) found.")

    # 2. Load only reference tables for FK resolution
    print("  Loading reference tables for FK resolution...")
    lookup_rows = fetch_lookup_tables(conn, table_meta)
    lookups, table_key_map = build_lookup_tables_from_rows(lookup_rows)
    print(f"  {len(lookups)} FK reference table(s) resolved from {len(lookup_rows)} lookup table(s).")

    # 3. Create index
    ensure_index(index_name, fresh=fresh)

    # 4. Index schema document
    schema_text = build_schema_document_from_rows(
        {t: [] for t in table_meta},  # schema only needs table/column names, not rows
        lookups, table_key_map
    )
    schema_chunks = schema_to_chunks(schema_text)
    print(f"\n  Indexing schema ({len(schema_chunks)} chunk(s))...")
    doc_id_offset = bulk_index(schema_chunks, index_name, start_id=0)
    total_chunks = doc_id_offset

    # 5. Stream and index each table
    skipped = []
    for tbl, meta in table_meta.items():
        row_count = meta['row_count']

        if row_count > MAX_ROWS_PER_TABLE:
            skipped.append((tbl, row_count))
            continue

        if row_count == 0:
            continue

        print(f"  {tbl}: ~{row_count:,} rows — streaming and indexing...")
        try:
            indexed = stream_and_index_table(conn, tbl, lookups, index_name,
                                             start_id=doc_id_offset, total_rows=row_count)
            doc_id_offset += indexed
            total_chunks += indexed
        except Exception as e:
            print(f"    Error processing {tbl}: {e}")

    conn.close()

    if skipped:
        print(f"\n  Skipped {len(skipped)} oversized table(s) (> {MAX_ROWS_PER_TABLE:,} rows):")
        for tbl, count in skipped:
            print(f"    - {tbl}: ~{count:,} rows  (raise MAX_ROWS_PER_TABLE to include)")

    print(f"\n  Done. {total_chunks} chunk(s) indexed into '{index_name}'.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    fresh = '--fresh' in sys.argv
    configs = get_db_configs()
    print(f"Processing {len(configs)} database(s)...")
    print(f"Row limit per table: {MAX_ROWS_PER_TABLE:,}  (MAX_ROWS_PER_TABLE)")
    if fresh:
        print("Mode: fresh (existing indexes will be deleted and recreated)")
    for config in configs:
        process_database(config, fresh=fresh)
    print(f"\nAll databases processed.")


if __name__ == '__main__':
    main()
