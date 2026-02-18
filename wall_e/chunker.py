"""
Shared chunking, embedding and indexing utilities.
Used by chunk_data_dir.py and chunk_db.py.
"""
import os
import re
import csv
import nltk
import tiktoken
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ── NLTK ─────────────────────────────────────────────────────────────────────
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize  # noqa: E402 (after download)

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE           = int(os.getenv("BATCH_SIZE", "32"))
CHUNK_MIN_TOKENS     = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX_TOKENS     = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
PARAGRAPH_BREAK_NEWLINES = int(os.getenv("PARAGRAPH_BREAK_NEWLINES", "2"))

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

# ── Clients ───────────────────────────────────────────────────────────────────
client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    use_ssl=False
)
model = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = tiktoken.get_encoding("cl100k_base")

# ── Index template ────────────────────────────────────────────────────────────
INDEX_BODY = {
    "settings": {
        "index.knn": True,
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "content_vector": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "lucene"
                }
            },
            "text":         {"type": "text"},
            "doc_name":     {"type": "keyword"},
            "doc_type":     {"type": "keyword"},
            "source_type":  {"type": "keyword"},
            "table_name":   {"type": "keyword"},
            "chunk_index":  {"type": "integer"},
            "source_file":  {"type": "keyword"},
            "page_numbers": {"type": "keyword"},
            "start_page":   {"type": "integer"},
            "end_page":     {"type": "integer"},
        }
    }
}


def ensure_index(index_name: str, fresh: bool = False):
    """
    Create the index if it doesn't exist.
    If fresh=True, delete and recreate it (clean slate for re-runs).
    """
    try:
        exists = client.indices.exists(index=index_name)
        if exists and fresh:
            print(f"Deleting existing index '{index_name}' (--fresh)...")
            client.indices.delete(index=index_name)
            exists = False
        if exists:
            print(f"Index '{index_name}' already exists. Using existing index.")
        else:
            print(f"Creating index '{index_name}'...")
            client.indices.create(index=index_name, body=INDEX_BODY)
            print("Index created.")
    except Exception as e:
        print(f"Warning: could not create index ({e}). Will attempt auto-creation on first insert.")


def bulk_index(chunks: list[dict], index_name: str, start_id: int = 0) -> int:
    """
    Embed a list of chunk dicts and bulk-insert them into OpenSearch.
    Returns the number of documents indexed.
    """
    total = len(chunks)
    doc_id = start_id
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c['text'] for c in batch]
        vectors = model.encode(texts, batch_size=BATCH_SIZE).tolist()

        actions = []
        for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
            actions.append({
                "_index": index_name,
                "_id": str(doc_id + offset),
                "_source": {
                    "text":          chunk['text'],
                    "content_vector": vector,
                    "doc_name":      chunk.get('doc_name', ''),
                    "doc_type":      chunk.get('doc_type', ''),
                    "source_type":   chunk.get('source_type', ''),
                    "table_name":    chunk.get('table_name', ''),
                    "chunk_index":   chunk.get('chunk_index', 0),
                    "source_file":   chunk.get('source_file', ''),
                    "page_numbers":  chunk.get('page_numbers', ''),
                    "start_page":    chunk.get('start_page', 0),
                    "end_page":      chunk.get('end_page', 0),
                }
            })
        helpers.bulk(client, actions)
        doc_id += len(actions)

    return total


# ── Name helpers ──────────────────────────────────────────────────────────────

def normalize_doc_key(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return re.sub(r"[^a-z0-9]+", "", base.lower())


def clean_doc_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    base = re.sub(r"\(\d+\)$", "", base).strip()
    base = base.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", base).strip()


def infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    if "lis" in lower or "interface" in lower:
        return "LIS Integration Guide"
    if "service manual" in lower or lower.endswith("_sm.pdf") or " sm.pdf" in lower:
        return "Service Manual"
    if "user manual" in lower or "manual" in lower:
        return "User Manual"
    return "Document"


def dir_to_index_name(dir_path: str) -> str:
    """e.g. wall_e/data/eqa-monthly  →  eqa_monthly_index"""
    basename = os.path.basename(dir_path.rstrip('/'))
    return re.sub(r'[^a-z0-9]+', '_', basename.lower()).strip('_') + '_index'


# ── FK resolution ─────────────────────────────────────────────────────────────

def build_lookup_tables(csv_dir: str) -> tuple[dict, dict]:
    """
    Scan CSVs under csv_dir. For every file with both 'id' and 'name' columns,
    build {id: name} lookup dicts keyed by table suffix.
    Returns (lookups, table_key_map).
    """
    lookups: dict = {}
    table_key_map: dict = {}

    for root, _dirs, files in os.walk(csv_dir):
        for fname in files:
            if not fname.lower().endswith('.csv'):
                continue
            path = os.path.join(root, fname)
            table_name = os.path.splitext(fname)[0]
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                    reader = csv.DictReader(fh)
                    if not reader.fieldnames:
                        continue
                    cols = [c.strip() for c in reader.fieldnames]
                    if 'id' not in cols or 'name' not in cols:
                        continue
                    id_to_name = {
                        (row.get('id') or '').strip(): (row.get('name') or '').strip()
                        for row in reader
                        if (row.get('id') or '').strip() and (row.get('name') or '').strip()
                    }
                    if not id_to_name:
                        continue
                    parts = table_name.split('_')
                    for i in range(len(parts)):
                        candidate = '_'.join(parts[i:])
                        if candidate not in lookups:
                            lookups[candidate] = id_to_name
                            table_key_map[candidate] = table_name
                            break
                    else:
                        lookups[table_name] = id_to_name
                        table_key_map[table_name] = table_name
            except Exception:
                continue

    return lookups, table_key_map


def build_lookup_tables_from_rows(tables: dict[str, list[dict]]) -> tuple[dict, dict]:
    """
    Build lookup tables from an in-memory dict of {table_name: [row_dicts]}.
    Used by chunk_db.py where rows come directly from Postgres.
    """
    lookups: dict = {}
    table_key_map: dict = {}

    for table_name, rows in tables.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        if 'id' not in cols or 'name' not in cols:
            continue
        id_to_name = {
            str(row.get('id', '')).strip(): str(row.get('name', '')).strip()
            for row in rows
            if row.get('id') and row.get('name')
        }
        if not id_to_name:
            continue
        parts = table_name.split('_')
        for i in range(len(parts)):
            candidate = '_'.join(parts[i:])
            if candidate not in lookups:
                lookups[candidate] = id_to_name
                table_key_map[candidate] = table_name
                break
        else:
            lookups[table_name] = id_to_name
            table_key_map[table_name] = table_name

    return lookups, table_key_map


def resolve_foreign_keys(row: dict, headers: list[str], lookups: dict) -> dict:
    resolved = {}
    skip = {'created_by', 'updated_by', 'approved_by', 'confirmed_by', 'reviewed_by', 'lab_user'}
    for col in headers:
        val = (row.get(col) or '').strip()
        if not val or val in ('--', '-----', '""', "''"):
            continue
        resolved[col] = val
        if col.endswith('_id'):
            fk_base = col[:-3]
            if fk_base in skip:
                continue
            lookup = lookups.get(fk_base)
            if lookup and val in lookup:
                resolved[f"{fk_base}_name"] = lookup[val]
    return resolved


def build_schema_document(csv_dir: str, lookups: dict, table_key_map: dict) -> str:
    lines = [
        "DATABASE SCHEMA AND TABLE RELATIONSHIPS",
        "=" * 50,
        "This document describes all data tables, their columns, "
        "and how they relate to each other via foreign keys.\n"
    ]
    for root, _dirs, files in os.walk(csv_dir):
        for fname in sorted(files):
            if not fname.lower().endswith('.csv'):
                continue
            path = os.path.join(root, fname)
            table_name = os.path.splitext(fname)[0]
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                    reader = csv.DictReader(fh)
                    if not reader.fieldnames:
                        continue
                    cols = [c.strip() for c in reader.fieldnames]
            except Exception:
                continue
            fk_descs = [
                f"  - {col} → references {table_key_map.get(col[:-3], col[:-3])}.id"
                for col in cols
                if col.endswith('_id') and col[:-3] in lookups
            ]
            lines.append(f"Table: {table_name}")
            lines.append(f"  Columns: {', '.join(cols)}")
            if fk_descs:
                lines.append("  Foreign keys:")
                lines.extend(fk_descs)
            lines.append("")
    return "\n".join(lines)


def build_schema_document_from_rows(tables: dict[str, list[dict]], lookups: dict, table_key_map: dict) -> str:
    """Build schema document from in-memory table data (used by chunk_db.py)."""
    lines = [
        "DATABASE SCHEMA AND TABLE RELATIONSHIPS",
        "=" * 50,
        "This document describes all data tables, their columns, "
        "and how they relate to each other via foreign keys.\n"
    ]
    for table_name, rows in sorted(tables.items()):
        cols = list(rows[0].keys()) if rows else []
        fk_descs = [
            f"  - {col} → references {table_key_map.get(col[:-3], col[:-3])}.id"
            for col in cols
            if col.endswith('_id') and col[:-3] in lookups
        ]
        lines.append(f"Table: {table_name}")
        lines.append(f"  Columns: {', '.join(cols)}")
        if fk_descs:
            lines.append("  Foreign keys:")
            lines.extend(fk_descs)
        lines.append("")
    return "\n".join(lines)


# ── PDF chunking ──────────────────────────────────────────────────────────────

def get_chunks_with_pages(pdf_path: str) -> list[dict]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages_data = [
        {'page_num': i + 1, 'text': page.extract_text()}
        for i, page in enumerate(reader.pages)
        if page.extract_text() and page.extract_text().strip()
    ]
    if not pages_data:
        raise ValueError("No text extracted from PDF.")

    paragraphs_with_pages = []
    for pd_ in pages_data:
        raw = re.split(f'\n{{{PARAGRAPH_BREAK_NEWLINES},}}', pd_['text'])
        for para in raw:
            para = para.strip()
            if para:
                paragraphs_with_pages.append({
                    'sentences': [s.strip() for s in sent_tokenize(para) if s.strip()],
                    'page_num': pd_['page_num']
                })

    chunks = []
    current_sents, current_tokens = [], 0
    start_page = end_page = None

    for pd_ in paragraphs_with_pages:
        for sentence in pd_['sentences']:
            stokens = len(tokenizer.encode(sentence))
            if start_page is None:
                start_page = end_page = pd_['page_num']
            end_page = max(end_page, pd_['page_num'])

            if current_tokens + stokens > CHUNK_MAX_TOKENS and current_sents:
                page_range = str(start_page) if start_page == end_page else f"{start_page}-{end_page}"
                chunks.append({'text': " ".join(current_sents), 'start_page': start_page,
                               'end_page': end_page, 'page_numbers': page_range})
                overlap_sents, overlap_tokens = [], 0
                for s in reversed(current_sents):
                    st = len(tokenizer.encode(s))
                    if overlap_tokens + st <= CHUNK_OVERLAP_TOKENS:
                        overlap_sents.insert(0, s)
                        overlap_tokens += st
                    else:
                        break
                current_sents, current_tokens = overlap_sents, overlap_tokens
                start_page = end_page = pd_['page_num']

            current_sents.append(sentence)
            current_tokens += stokens

    if current_sents:
        page_range = str(start_page) if start_page == end_page else f"{start_page}-{end_page}"
        chunks.append({'text': " ".join(current_sents), 'start_page': start_page,
                       'end_page': end_page, 'page_numbers': page_range})
    return chunks


def process_pdf(pdf_path: str, doc_name: str, doc_type: str) -> list[dict]:
    chunks = get_chunks_with_pages(pdf_path)
    return [
        {
            'text': c['text'], 'doc_name': doc_name, 'doc_type': doc_type,
            'source_type': 'pdf', 'table_name': '', 'chunk_index': i,
            'source_file': os.path.basename(pdf_path),
            'start_page': c['start_page'], 'end_page': c['end_page'],
            'page_numbers': c['page_numbers']
        }
        for i, c in enumerate(chunks)
    ]


# ── CSV chunking ──────────────────────────────────────────────────────────────

def get_csv_chunks(csv_path: str, lookups: dict | None = None) -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if lookups is None:
        lookups = {}

    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    chunks = []

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return chunks
        headers = [c.strip() for c in reader.fieldnames]
        header_text = f"Table: {table_name}\nColumns: {', '.join(headers)}\n\n"
        header_tokens = len(tokenizer.encode(header_text))
        current_rows, current_tokens, start_row = [], header_tokens, 1

        for row_num, row in enumerate(reader, start=1):
            resolved = resolve_foreign_keys(row, headers, lookups)
            if not resolved:
                continue
            row_text = ", ".join(f"{k}={v}" for k, v in resolved.items())
            row_tokens = len(tokenizer.encode(row_text))

            if current_tokens + row_tokens > CHUNK_MAX_TOKENS and current_rows:
                chunks.append({
                    'text': header_text + "\n".join(current_rows),
                    'start_row': start_row, 'end_row': row_num - 1,
                    'row_range': f"rows {start_row}-{row_num - 1}"
                })
                current_rows, current_tokens, start_row = [], header_tokens, row_num

            current_rows.append(row_text)
            current_tokens += row_tokens

        if current_rows:
            end_row = start_row + len(current_rows) - 1
            chunks.append({
                'text': header_text + "\n".join(current_rows),
                'start_row': start_row, 'end_row': end_row,
                'row_range': f"rows {start_row}-{end_row}"
            })

    return chunks


def process_csv(csv_path: str, lookups: dict | None = None) -> list[dict]:
    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    return [
        {
            'text': c['text'], 'doc_name': table_name, 'doc_type': 'CSV Data',
            'source_type': 'csv', 'table_name': table_name, 'chunk_index': i,
            'source_file': os.path.basename(csv_path),
            'start_page': c['start_row'], 'end_page': c['end_row'],
            'page_numbers': c['row_range']
        }
        for i, c in enumerate(get_csv_chunks(csv_path, lookups=lookups))
    ]


def process_rows(table_name: str, rows: list[dict], lookups: dict | None = None) -> list[dict]:
    """
    Chunk an in-memory list of row dicts (used by chunk_db.py).
    Equivalent to process_csv but works on rows already fetched from Postgres.
    """
    if lookups is None:
        lookups = {}
    if not rows:
        return []

    headers = list(rows[0].keys())
    header_text = f"Table: {table_name}\nColumns: {', '.join(headers)}\n\n"
    header_tokens = len(tokenizer.encode(header_text))
    current_rows, current_tokens, start_row = [], header_tokens, 1
    chunks = []

    for row_num, row in enumerate(rows, start=1):
        str_row = {k: str(v) if v is not None else '' for k, v in row.items()}
        resolved = resolve_foreign_keys(str_row, headers, lookups)
        if not resolved:
            continue
        row_text = ", ".join(f"{k}={v}" for k, v in resolved.items())
        row_tokens = len(tokenizer.encode(row_text))

        if current_tokens + row_tokens > CHUNK_MAX_TOKENS and current_rows:
            chunks.append({
                'text': header_text + "\n".join(current_rows),
                'start_row': start_row, 'end_row': row_num - 1,
                'row_range': f"rows {start_row}-{row_num - 1}"
            })
            current_rows, current_tokens, start_row = [], header_tokens, row_num

        current_rows.append(row_text)
        current_tokens += row_tokens

    if current_rows:
        end_row = start_row + len(current_rows) - 1
        chunks.append({
            'text': header_text + "\n".join(current_rows),
            'start_row': start_row, 'end_row': end_row,
            'row_range': f"rows {start_row}-{end_row}"
        })

    return [
        {
            'text': c['text'], 'doc_name': table_name, 'doc_type': 'CSV Data',
            'source_type': 'csv', 'table_name': table_name, 'chunk_index': i,
            'source_file': table_name, 'start_page': c['start_row'],
            'end_page': c['end_row'], 'page_numbers': c['row_range']
        }
        for i, c in enumerate(chunks)
    ]


# ── Schema chunk helpers ──────────────────────────────────────────────────────

def schema_to_chunks(schema_text: str) -> list[dict]:
    """Split a schema document into indexable chunks."""
    schema_tokens = len(tokenizer.encode(schema_text))
    if schema_tokens <= CHUNK_MAX_TOKENS * 3:
        texts = [schema_text]
    else:
        sections = schema_text.split('\nTable: ')
        current = sections[0]
        texts = []
        for section in sections[1:]:
            full = '\nTable: ' + section
            if len(tokenizer.encode(current + full)) > CHUNK_MAX_TOKENS:
                texts.append(current)
                current = "DATABASE SCHEMA (continued)\n" + full
            else:
                current += full
        if current:
            texts.append(current)

    return [
        {
            'text': t, 'doc_name': 'database_schema', 'doc_type': 'Schema',
            'source_type': 'schema', 'table_name': '', 'chunk_index': i,
            'source_file': '_schema', 'page_numbers': 'schema',
            'start_page': 0, 'end_page': 0
        }
        for i, t in enumerate(texts)
    ]
