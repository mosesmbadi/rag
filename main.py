# This code will chunk given data and call embedding code to embedd the data.
import sys
import os
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 1. SETUP: Connect to OpenSearch and load the Embedding Model
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
            "text": {"type": "text"}
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

# 4. EMBEDDING & STORAGE: Process the PDF
pdf_path = "/home/mbadi/Desktop/work/chemlabs/chemlabs_rag/data/humacount_5d_sm.pdf"

try:
    print(f"Processing PDF: {pdf_path}")
    pdf_chunks = get_chunks(pdf_path)
    print(f"Created {len(pdf_chunks)} chunks from the PDF.")
    
    print("Embedding and indexing chunks...")
    for i, chunk in enumerate(pdf_chunks):
        vector = model.encode(chunk).tolist()
        doc = {
            "text": chunk,
            "content_vector": vector
        }
        client.index(index=index_name, body=doc, id=str(i))
        
        if (i + 1) % 10 == 0:
            print(f"Indexed {i + 1}/{len(pdf_chunks)} chunks...")
    
    print(f"\n✓ Successfully indexed {len(pdf_chunks)} chunks from the PDF.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error processing PDF: {e}")
    sys.exit(1)