# This code will chunk given data and call embedding code to embedd the data.
import sys
import os
import re
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 1. SETUP: Connect to vector-based OpenSearch and load the Embedding Model
# No authentication needed since DISABLE_SECURITY_PLUGIN=true
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    use_ssl=False
)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Generates 384-dimension vectors

# 2. CHUNKING: Read PDF and split into snippets
def get_chunks(pdf_path, chunk_size=800, overlap=200):
    """
    Extract text from PDF and split into overlapping chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each chunk in characters (increased from 500 to 800)
        overlap: Number of characters to overlap between chunks (increased from 50 to 200)
    
    Returns:
        List of text chunks
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF")
    
    # Create overlapping chunks for better context preservation
    chunks = []
    step_size = chunk_size - overlap
    
    for i in range(0, len(text), step_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def process_pdf(pdf_path, doc_name, doc_type):
    """
    Process a single PDF and return chunks with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        doc_name: Name/identifier of the document (e.g., "HumaCount 5D")
        doc_type: Type of document (e.g., "Service Manual", "User Manual")
    
    Returns:
        List of dictionaries with 'text' and metadata
    """
    chunks = get_chunks(pdf_path)
    
    # Add metadata to each chunk
    chunks_with_metadata = []
    for i, chunk_text in enumerate(chunks):
        chunks_with_metadata.append({
            'text': chunk_text,
            'doc_name': doc_name,
            'doc_type': doc_type,
            'chunk_index': i,
            'source_file': os.path.basename(pdf_path)
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
            "source_file": {"type": "keyword"}
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
    all_chunks = []
    doc_id = 0
    
    for doc in documents:
        print(f"\nProcessing: {doc['name']} ({doc['type']})")
        print(f"File: {doc['path']}")
        
        chunks_with_metadata = process_pdf(doc['path'], doc['name'], doc['type'])
        print(f"Created {len(chunks_with_metadata)} chunks")
        
        all_chunks.extend(chunks_with_metadata)
    
    print(f"\n{'='*80}")
    print(f"Total chunks across all documents: {len(all_chunks)}")
    print("Embedding and indexing chunks...")
    
    for i, chunk_data in enumerate(all_chunks):
        vector = model.encode(chunk_data['text']).tolist()
        doc = {
            "text": chunk_data['text'],
            "content_vector": vector,
            "doc_name": chunk_data['doc_name'],
            "doc_type": chunk_data['doc_type'],
            "chunk_index": chunk_data['chunk_index'],
            "source_file": chunk_data['source_file']
        }
        client.index(index=index_name, body=doc, id=str(doc_id))
        doc_id += 1
        
        if (i + 1) % 10 == 0:
            print(f"Indexed {i + 1}/{len(all_chunks)} chunks...")
    
    print(f"\n✓ Successfully indexed {len(all_chunks)} chunks from {len(documents)} document(s).")
    
    # Print summary by document
    print(f"\nDocument summary:")
    for doc in documents:
        doc_chunks = [c for c in all_chunks if c['doc_name'] == doc['name'] and c['doc_type'] == doc['type']]
        print(f"  - {doc['name']} ({doc['type']}): {len(doc_chunks)} chunks")
        
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error processing PDFs: {e}")
    sys.exit(1)