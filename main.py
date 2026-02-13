# This code will chunk given data and call embedding code to embedd the data.
import sys
import os
import re
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import csv
import json
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Download NLTK data for sentence tokenization (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# 1. SETUP: Connect to vector-based OpenSearch and load the Embedding Model
# No authentication needed since DISABLE_SECURITY_PLUGIN=true
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    use_ssl=False
)

# Load embedding model from environment variable
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
model = SentenceTransformer(EMBEDDING_MODEL)  # Generates 384-dimension vectors
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Chunking configuration
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
PARAGRAPH_BREAK_NEWLINES = int(os.getenv("PARAGRAPH_BREAK_NEWLINES", "2"))

# CSV chunking configuration
CSV_ROWS_PER_CHUNK = int(os.getenv("CSV_ROWS_PER_CHUNK", "50"))

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer


# ── FK RESOLUTION: Build lookup tables from reference CSVs ──────────────────

def build_lookup_tables(csv_dir):
    """
    Scan all CSV files under csv_dir. For every file that has both 'id' and
    'name' columns, build a lookup dict  {id_string: name_string}.

    Returns two dicts:
      - lookups:        {table_key: {id: name}}   e.g. {"program": {"4": "HuQAS"}}
      - table_key_map:  {table_key: full_table_name}  e.g. {"program": "catalog_program"}
    """
    lookups = {}          # table_key → {id → name}
    table_key_map = {}    # table_key → original table name (without ext)

    for root, _dirs, files in os.walk(csv_dir):
        for fname in files:
            if not fname.lower().endswith('.csv'):
                continue
            path = os.path.join(root, fname)
            table_name = os.path.splitext(fname)[0]          # e.g. catalog_program

            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                    reader = csv.DictReader(fh)
                    if not reader.fieldnames:
                        continue
                    cols = [c.strip() for c in reader.fieldnames]
                    if 'id' not in cols or 'name' not in cols:
                        continue

                    id_to_name = {}
                    for row in reader:
                        rid = (row.get('id') or '').strip()
                        rname = (row.get('name') or '').strip()
                        if rid and rname:
                            id_to_name[rid] = rname

                    if not id_to_name:
                        continue

                    # Derive a short key by stripping common prefixes
                    # catalog_program → program, labs_lab → lab, test_events_event → event
                    parts = table_name.split('_')
                    # Try progressively shorter suffixes to avoid collisions
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
                continue  # skip unreadable files

    return lookups, table_key_map


def resolve_foreign_keys(row, headers, lookups):
    """
    For every column ending with '_id', try to resolve it via the lookup
    tables.  Returns a *new* dict with both the original id and the resolved
    human-readable name.

    Example:
        program_id=4  →  program_id=4, program_name=HuQAS
    """
    resolved = {}
    for col in headers:
        val = (row.get(col) or '').strip()
        if not val or val in ('--', '-----', '""', "''"):
            continue
        resolved[col] = val

        if col.endswith('_id'):
            fk_base = col[:-3]  # program_id → program
            # Skip audit columns – they don't add useful search context
            if fk_base in ('created_by', 'updated_by', 'approved_by',
                           'confirmed_by', 'reviewed_by', 'lab_user'):
                continue
            # Look up by the FK base name (e.g. "program", "lab", "analyte")
            lookup = lookups.get(fk_base)
            if lookup and val in lookup:
                resolved[f"{fk_base}_name"] = lookup[val]
    return resolved


def build_schema_document(csv_dir, lookups, table_key_map):
    """
    Generate a text document describing every CSV table, its columns, and
    which FK columns link to which reference tables.  This chunk is indexed
    so the LLM can reason about cross-table relationships.
    """
    lines = [
        "DATABASE SCHEMA AND TABLE RELATIONSHIPS",
        "=" * 50,
        "This document describes all data tables, their columns, "
        "and how they relate to each other via foreign keys.\n"
    ]

    # Build reverse map: table_key → full_table_name
    key_to_full = {v: k for k, v in table_key_map.items()}  # full→key

    tables_info = []  # (table_name, cols, fk_descriptions)
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

            fk_descs = []
            for col in cols:
                if col.endswith('_id'):
                    fk_base = col[:-3]
                    if fk_base in lookups:
                        ref_table = table_key_map.get(fk_base, fk_base)
                        fk_descs.append(f"  - {col} → references {ref_table}.id (lookup by name)")
                    else:
                        fk_descs.append(f"  - {col} → references related table")

            tables_info.append((table_name, cols, fk_descs))

    for table_name, cols, fk_descs in tables_info:
        lines.append(f"Table: {table_name}")
        lines.append(f"  Columns: {', '.join(cols)}")
        if fk_descs:
            lines.append("  Foreign keys:")
            lines.extend(fk_descs)
        lines.append("")

    return "\n".join(lines)

# 2. CHUNKING: Read PDF and split into paragraph-aware, token-limited chunks with page metadata
def get_chunks_with_pages(pdf_path, min_tokens=CHUNK_MIN_TOKENS, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """
    Extract text from PDF and split into paragraph-aware chunks with page metadata.
    
    This approach:
    - Tracks page numbers for each chunk
    - Respects paragraph boundaries (detects via multiple newlines)
    - Groups sentences within paragraphs
    - Uses token-based sizing for better LLM compatibility
    - Maintains context with overlapping chunks
    
    Args:
        pdf_path: Path to the PDF file
        min_tokens: Minimum tokens per chunk (default: 300)
        max_tokens: Maximum tokens per chunk (default: 500)
        overlap_tokens: Number of tokens to overlap between chunks (default: 50)
    
    Returns:
        List of dictionaries with 'text', 'start_page', 'end_page', 'page_numbers'
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    
    # Extract text page by page with page tracking
    pages_data = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages_data.append({
                'page_num': page_num,
                'text': page_text
            })
    
    if not pages_data:
        raise ValueError("No text could be extracted from the PDF")
    
    # Detect paragraphs across all pages
    # A paragraph is a group of sentences separated by multiple newlines
    paragraph_break_pattern = '\n' * PARAGRAPH_BREAK_NEWLINES
    
    paragraphs_with_pages = []
    for page_data in pages_data:
        page_num = page_data['page_num']
        text = page_data['text']
        
        # Split by paragraph breaks
        raw_paragraphs = re.split(f'\n{{{PARAGRAPH_BREAK_NEWLINES},}}', text)
        
        for para_text in raw_paragraphs:
            para_text = para_text.strip()
            if para_text:
                # Split paragraph into sentences
                sentences = sent_tokenize(para_text)
                paragraphs_with_pages.append({
                    'sentences': [s.strip() for s in sentences if s.strip()],
                    'page_num': page_num
                })
    
    # Now create chunks respecting paragraph boundaries
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    current_chunk_start_page = None
    current_chunk_end_page = None
    
    for para_data in paragraphs_with_pages:
        sentences = para_data['sentences']
        page_num = para_data['page_num']
        
        for sentence in sentences:
            sentence_tokens = len(tokenizer.encode(sentence))
            
            # Initialize page tracking for first sentence
            if current_chunk_start_page is None:
                current_chunk_start_page = page_num
                current_chunk_end_page = page_num
            
            # Update end page
            current_chunk_end_page = max(current_chunk_end_page, page_num)
            
            # If adding this sentence would exceed max_tokens, save current chunk
            if current_chunk_tokens + sentence_tokens > max_tokens and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    'text': chunk_text,
                    'start_page': current_chunk_start_page,
                    'end_page': current_chunk_end_page,
                    'page_numbers': f"{current_chunk_start_page}" if current_chunk_start_page == current_chunk_end_page else f"{current_chunk_start_page}-{current_chunk_end_page}"
                })
                
                # Create overlap: keep last few sentences for context
                overlap_sentences = []
                overlap_tokens_count = 0
                for sent in reversed(current_chunk_sentences):
                    sent_tokens = len(tokenizer.encode(sent))
                    if overlap_tokens_count + sent_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens_count += sent_tokens
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = overlap_tokens_count
                # Reset page tracking for new chunk
                current_chunk_start_page = page_num
                current_chunk_end_page = page_num
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
    
    # Add the last chunk if it has content
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        if current_chunk_tokens >= min_tokens or not chunks:
            chunks.append({
                'text': chunk_text,
                'start_page': current_chunk_start_page,
                'end_page': current_chunk_end_page,
                'page_numbers': f"{current_chunk_start_page}" if current_chunk_start_page == current_chunk_end_page else f"{current_chunk_start_page}-{current_chunk_end_page}"
            })
    
    return chunks

def process_pdf(pdf_path, doc_name, doc_type):
    """
    Process a single PDF and return chunks with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        doc_name: Name/identifier of the document (e.g., "HumaCount 5D")
        doc_type: Type of document (e.g., "Service Manual", "User Manual")
    
    Returns:
        List of dictionaries with 'text' and metadata including page numbers
    """
    chunks = get_chunks_with_pages(pdf_path)
    
    # Add metadata to each chunk
    chunks_with_metadata = []
    for i, chunk_data in enumerate(chunks):
        chunks_with_metadata.append({
            'text': chunk_data['text'],
            'doc_name': doc_name,
            'doc_type': doc_type,
            'source_type': 'pdf',
            'table_name': '',
            'chunk_index': i,
            'source_file': os.path.basename(pdf_path),
            'start_page': chunk_data['start_page'],
            'end_page': chunk_data['end_page'],
            'page_numbers': chunk_data['page_numbers']
        })
    
    return chunks_with_metadata


# 2b. CHUNKING: Read CSV and split into token-limited chunks with row metadata
def get_csv_chunks(csv_path, max_tokens=CHUNK_MAX_TOKENS, lookups=None):
    """
    Read a CSV file and create denormalized text chunks.

    Foreign key columns (ending in _id) are resolved to human-readable names
    using the lookup tables, making the chunks semantically searchable.

    Args:
        csv_path: Path to the CSV file
        max_tokens: Maximum tokens per chunk
        lookups: dict of {fk_base: {id: name}} for FK resolution

    Returns:
        List of dicts with 'text', 'start_row', 'end_row', 'row_range'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

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

        current_rows = []
        current_tokens = header_tokens
        start_row = 1

        for row_num, row in enumerate(reader, start=1):
            # Resolve FKs to human-readable names
            resolved = resolve_foreign_keys(row, headers, lookups)
            if not resolved:
                continue

            # Build compact key=value text with resolved names
            row_text = ", ".join(f"{k}={v}" for k, v in resolved.items())
            row_tokens = len(tokenizer.encode(row_text))

            # If adding this row would exceed limit, finalize current chunk
            if current_tokens + row_tokens > max_tokens and current_rows:
                chunk_text = header_text + "\n".join(current_rows)
                chunks.append({
                    'text': chunk_text,
                    'start_row': start_row,
                    'end_row': row_num - 1,
                    'row_range': f"rows {start_row}-{row_num - 1}"
                })
                current_rows = []
                current_tokens = header_tokens
                start_row = row_num

            current_rows.append(row_text)
            current_tokens += row_tokens

        # Final chunk
        if current_rows:
            end_row = start_row + len(current_rows) - 1
            chunk_text = header_text + "\n".join(current_rows)
            chunks.append({
                'text': chunk_text,
                'start_row': start_row,
                'end_row': end_row,
                'row_range': f"rows {start_row}-{end_row}"
            })

    return chunks


def process_csv(csv_path, lookups=None):
    """
    Process a single CSV file and return chunks with metadata.

    Args:
        csv_path: Path to the CSV file
        lookups: dict of {fk_base: {id: name}} for FK resolution

    Returns:
        List of dicts with 'text' and metadata including row range
    """
    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    chunks = get_csv_chunks(csv_path, lookups=lookups)

    chunks_with_metadata = []
    for i, chunk_data in enumerate(chunks):
        chunks_with_metadata.append({
            'text': chunk_data['text'],
            'doc_name': table_name,
            'doc_type': 'CSV Data',
            'source_type': 'csv',
            'table_name': table_name,
            'chunk_index': i,
            'source_file': os.path.basename(csv_path),
            'start_page': chunk_data['start_row'],
            'end_page': chunk_data['end_row'],
            'page_numbers': chunk_data['row_range']
        })

    return chunks_with_metadata


# 3. INDEXING: Create or use existing OpenSearch index with k-NN enabled
index_name = "my_pdf_index"
index_body = {
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
            "text": {"type": "text"},
            "doc_name": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "source_type": {"type": "keyword"},  # "pdf" or "csv"
            "table_name": {"type": "keyword"},   # CSV table name
            "chunk_index": {"type": "integer"},
            "source_file": {"type": "keyword"},
            "page_numbers": {"type": "keyword"},  # Pages ("5" or "5-7") or rows ("rows 1-50")
            "start_page": {"type": "integer"},    # Page or row start
            "end_page": {"type": "integer"}       # Page or row end
        }
    }
}

# Check if index exists and create if needed
try:
    if client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Using existing index.")
    else:
        print(f"Creating index '{index_name}'...")
        client.indices.create(index=index_name, body=index_body)
        print("Index created successfully.")
except Exception as e:
    print(f"Warning: Could not create index explicitly ({e}). Will try auto-creation on first document insert.")

# 4. EMBEDDING & STORAGE: Process PDFs
def normalize_doc_key(filename):
    base = os.path.splitext(filename)[0]
    return re.sub(r"[^a-z0-9]+", "", base.lower())


def clean_doc_name(filename):
    base = os.path.splitext(filename)[0]
    base = re.sub(r"\(\d+\)$", "", base).strip()
    base = base.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", base).strip()


def infer_doc_type(filename):
    lower = filename.lower()
    if "lis" in lower or "interface" in lower:
        return "LIS Integration Guide"
    if "service manual" in lower or lower.endswith("_sm.pdf") or " sm.pdf" in lower:
        return "Service Manual"
    if "user manual" in lower or "manual" in lower:
        return "User Manual"
    return "Document"


# Data directory from env, defaulting to ./data relative to this script
_default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
data_dir = os.getenv("DATA_DIR", _default_data_dir)
if not os.path.isabs(data_dir):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)

if not os.path.isdir(data_dir):
    print(f"Error: Data directory not found: {data_dir}")
    sys.exit(1)

# ── Build FK lookup tables from reference CSVs ──────────────────────────────
print("Building FK lookup tables from reference CSVs...")
lookups, table_key_map = build_lookup_tables(data_dir)
print(f"  Loaded {len(lookups)} reference tables: {', '.join(sorted(table_key_map.values()))}")

# Discover all supported files recursively (PDFs + CSVs)
SUPPORTED_EXTENSIONS = {'.pdf', '.csv'}
documents = []
seen = set()

for root, dirs, files in os.walk(data_dir):
    for file_name in sorted(files):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        key = normalize_doc_key(file_name)
        if key in seen:
            continue
        seen.add(key)

        file_path = os.path.join(root, file_name)

        if ext == '.pdf':
            documents.append({
                "path": file_path,
                "name": clean_doc_name(file_name),
                "type": infer_doc_type(file_name),
                "file_type": "pdf"
            })
        elif ext == '.csv':
            table_name = os.path.splitext(file_name)[0]
            documents.append({
                "path": file_path,
                "name": table_name,
                "type": "CSV Data",
                "file_type": "csv"
            })

try:
    doc_id = 0
    total_chunks = 0
    doc_chunk_counts = {}
    batch_size = BATCH_SIZE

    # ── Index schema document first ─────────────────────────────────────────
    schema_text = build_schema_document(data_dir, lookups, table_key_map)
    if schema_text:
        print("\nIndexing database schema & relationships document...")
        schema_vector = model.encode(schema_text).tolist()
        # The schema can be long; split if needed, but usually one chunk is fine
        schema_chunks = []
        schema_tokens = len(tokenizer.encode(schema_text))
        if schema_tokens <= CHUNK_MAX_TOKENS * 3:
            schema_chunks = [schema_text]
        else:
            # Split into sections by table
            sections = schema_text.split('\nTable: ')
            current = sections[0]  # header
            for section in sections[1:]:
                section_full = '\nTable: ' + section
                if len(tokenizer.encode(current + section_full)) > CHUNK_MAX_TOKENS:
                    schema_chunks.append(current)
                    current = "DATABASE SCHEMA (continued)\n" + section_full
                else:
                    current += section_full
            if current:
                schema_chunks.append(current)

        for si, s_text in enumerate(schema_chunks):
            s_vector = model.encode(s_text).tolist()
            client.index(
                index=index_name,
                id=str(doc_id),
                body={
                    "text": s_text,
                    "content_vector": s_vector,
                    "doc_name": "database_schema",
                    "doc_type": "Schema",
                    "source_type": "schema",
                    "table_name": "",
                    "chunk_index": si,
                    "source_file": "_schema",
                    "page_numbers": "schema",
                    "start_page": 0,
                    "end_page": 0
                }
            )
            doc_id += 1
        total_chunks += len(schema_chunks)
        print(f"  Indexed {len(schema_chunks)} schema chunk(s)")

    for doc in documents:
        print(f"\nProcessing: {doc['name']} ({doc['type']})")
        print(f"File: {doc['path']}")

        if doc['file_type'] == 'pdf':
            chunks_with_metadata = process_pdf(doc['path'], doc['name'], doc['type'])
        elif doc['file_type'] == 'csv':
            chunks_with_metadata = process_csv(doc['path'], lookups=lookups)
        else:
            print(f"  Skipping unsupported file type: {doc['path']}")
            continue
        chunk_count = len(chunks_with_metadata)
        total_chunks += chunk_count
        doc_chunk_counts[(doc['name'], doc['type'])] = chunk_count
        print(f"Created {chunk_count} chunks")

        print("Embedding and indexing chunks...")
        for start in range(0, chunk_count, batch_size):
            batch = chunks_with_metadata[start:start + batch_size]
            texts = [item['text'] for item in batch]
            vectors = model.encode(texts, batch_size=batch_size).tolist()

            actions = []
            for offset, (chunk_data, vector) in enumerate(zip(batch, vectors)):
                indexed_doc = {
                    "text": chunk_data['text'],
                    "content_vector": vector,
                    "doc_name": chunk_data['doc_name'],
                    "doc_type": chunk_data['doc_type'],
                    "source_type": chunk_data.get('source_type', 'pdf'),
                    "table_name": chunk_data.get('table_name', ''),
                    "chunk_index": chunk_data['chunk_index'],
                    "source_file": chunk_data['source_file'],
                    "page_numbers": chunk_data['page_numbers'],
                    "start_page": chunk_data['start_page'],
                    "end_page": chunk_data['end_page']
                }
                actions.append({
                    "_index": index_name,
                    "_id": str(doc_id + offset),
                    "_source": indexed_doc
                })

            helpers.bulk(client, actions)
            doc_id += len(actions)

        print(f"Indexed {chunk_count} chunks for {doc['name']}")
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully indexed {total_chunks} chunks from {len(documents)} document(s).")
    
    # Print summary by document
    print(f"\nDocument summary:")
    for doc in documents:
        count = doc_chunk_counts.get((doc['name'], doc['type']), 0)
        print(f"  - {doc['name']} ({doc['type']}): {count} chunks")
        
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error processing documents: {e}")
    sys.exit(1)