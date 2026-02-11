# This code will chunk given data and call embedding code to embedd the data.
import sys
import os
import re
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

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

# Initialize tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

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
            'chunk_index': i,
            'source_file': os.path.basename(pdf_path),
            'start_page': chunk_data['start_page'],
            'end_page': chunk_data['end_page'],
            'page_numbers': chunk_data['page_numbers']
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
            "chunk_index": {"type": "integer"},
            "source_file": {"type": "keyword"},
            "page_numbers": {"type": "keyword"},  # Display format: "5" or "5-7"
            "start_page": {"type": "integer"},    # For filtering/sorting
            "end_page": {"type": "integer"}       # For range queries
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


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
pdf_files = sorted(
    file_name for file_name in os.listdir(data_dir)
    if file_name.lower().endswith(".pdf")
)

documents = []
seen = set()

for file_name in pdf_files:
    key = normalize_doc_key(file_name)
    if key in seen:
        continue
    seen.add(key)

    documents.append({
        "path": os.path.join(data_dir, file_name),
        "name": clean_doc_name(file_name),
        "type": infer_doc_type(file_name)
    })

try:
    doc_id = 0
    total_chunks = 0
    doc_chunk_counts = {}
    batch_size = BATCH_SIZE

    for doc in documents:
        print(f"\nProcessing: {doc['name']} ({doc['type']})")
        print(f"File: {doc['path']}")
 
        chunks_with_metadata = process_pdf(doc['path'], doc['name'], doc['type'])
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
    print(f"Error processing PDFs: {e}")
    sys.exit(1)