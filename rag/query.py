import sys
import os
import re
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to OpenSearch (same setup as main.py)
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', '9200'))
client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    use_ssl=False
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Resolve which OpenSearch indices to search.
# INDEX_NAMES takes priority: comma-separated list of index names.
# Falls back to deriving a single index name from DATA_DIR basename.
_index_names_env = os.getenv("INDEX_NAMES", "").strip()
if _index_names_env:
    index_name = ",".join(n.strip() for n in _index_names_env.split(",") if n.strip())
else:
    _default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    _data_dir = os.getenv("DATA_DIR", _default_data_dir)
    _dir_basename = os.path.basename(_data_dir.rstrip('/'))
    index_name = re.sub(r'[^a-z0-9]+', '_', _dir_basename.lower()).strip('_') + '_index'

print(f"Searching index/indices: {index_name}")


def normalize_doc_key(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def tokenize_for_match(text):
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    tokens = [token for token in cleaned.split() if token]
    stopwords = {
        "user",
        "manual",
        "service",
        "lis",
        "interface",
        "guide",
        "version",
        "ver",
        "v"
    }
    return [token for token in tokens if token not in stopwords]


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


def build_document_catalog():
    _default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    data_dir = os.getenv("DATA_DIR", _default_data_dir)
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
    if not os.path.isdir(data_dir):
        return []

    catalog = []
    seen = set()

    for root, dirs, files in os.walk(data_dir):
        for file_name in sorted(files):
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in ('.pdf', '.csv'):
                continue

            key = normalize_doc_key(os.path.splitext(file_name)[0])
            if key in seen:
                continue
            seen.add(key)

            if ext == '.pdf':
                catalog.append({
                    "doc_name": clean_doc_name(file_name),
                    "doc_type": infer_doc_type(file_name),
                    "source_file": file_name,
                    "source_type": "pdf",
                    "match_tokens": tokenize_for_match(clean_doc_name(file_name))
                })
            elif ext == '.csv':
                table_name = os.path.splitext(file_name)[0]
                catalog.append({
                    "doc_name": table_name,
                    "doc_type": "CSV Data",
                    "source_file": file_name,
                    "source_type": "csv",
                    "match_tokens": tokenize_for_match(table_name.replace('_', ' '))
                })

    return catalog


DOCUMENT_CATALOG = build_document_catalog()


def infer_doc_filter_from_question(question):
    if not DOCUMENT_CATALOG:
        return None

    normalized_question = normalize_doc_key(question)
    question_tokens = set(tokenize_for_match(question))
    best_match = None
    best_score = 0

    for entry in DOCUMENT_CATALOG:
        doc_key = normalize_doc_key(entry["doc_name"])
        if not doc_key:
            continue
        if doc_key in normalized_question:
            score = len(doc_key)
            if score > best_score:
                best_score = score
                best_match = entry

        if question_tokens and entry["match_tokens"]:
            overlap = question_tokens.intersection(entry["match_tokens"])
            if overlap:
                score = len(overlap) * 10 + sum(len(token) for token in overlap)
                if score > best_score:
                    best_score = score
                    best_match = entry

    if best_match:
        return {
            "doc_name": best_match["doc_name"],
            "doc_type": best_match["doc_type"],
            "source_file": best_match["source_file"]
        }

    return None

# Minimum relevance score — results below this are considered noise
MIN_RELEVANCE_SCORE = float(os.getenv('MIN_RELEVANCE_SCORE', '0.5'))

# Load LLM configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
LLM_MODEL = os.getenv('LLM_MODEL_CPU_FALLBACK', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
MAX_ANSWER_LENGTH = int(os.getenv('MAX_ANSWER_LENGTH', '500'))

# Initialize Gemini if needed
gemini_model = None
if LLM_PROVIDER == 'gemini':
    try:
        from google import genai
        from google.genai import types
        if GEMINI_API_KEY:
            client_genai = genai.Client(api_key=GEMINI_API_KEY)
            gemini_model = client_genai
            print(f"Using Gemini API ({GEMINI_MODEL})")
        else:
            print("Warning: GEMINI_API_KEY not set in .env file")
            LLM_PROVIDER = 'local'
    except ImportError:
        print("Warning: google-genai not installed. Install with: pip install google-genai")
        LLM_PROVIDER = 'local'

# Initialize local LLM (lazy loading)
llm_tokenizer = None
llm_model = None

def load_llm():
    """Load the local LLM model and tokenizer (only when needed)"""
    global llm_tokenizer, llm_model
    
    if llm_model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"Loading local LLM model: {LLM_MODEL}...")
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print("Local LLM loaded successfully!")

def search_pdf(query, k=10, use_hybrid=True, doc_filter=None):
    """
    Search for relevant chunks from the PDF based on the query.
    
    Args:
        query: The question or search query
        k: Number of top results to return (default: 10)
        use_hybrid: Use hybrid search combining keyword + semantic (default: True)
        doc_filter: Optional dict to filter by document {'doc_name': 'HumaCount 5D', 'doc_type': 'Service Manual'}
    
    Returns:
        List of dictionaries with 'text', 'score', and metadata for each result
    """
    # Convert query to the same vector space as the indexed chunks
    query_vector = embedding_model.encode(query).tolist()

    # Build the base query
    if use_hybrid:
        # Hybrid search: combine semantic (kNN) and keyword (BM25) search
        base_query = {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "content_vector": {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "boost": 0.5  # Lower weight for keyword match
                            }
                        }
                    }
                ]
            }
        }
    else:
        # Pure semantic search
        base_query = {
            "knn": {
                "content_vector": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    
    # Add document filters if specified
    if doc_filter:
        filter_clauses = []
        if 'doc_name' in doc_filter:
            filter_clauses.append({"term": {"doc_name": doc_filter['doc_name']}})
        if 'doc_type' in doc_filter:
            filter_clauses.append({"term": {"doc_type": doc_filter['doc_type']}})
        if 'source_file' in doc_filter:
            filter_clauses.append({"term": {"source_file": doc_filter['source_file']}})
        
        if filter_clauses:
            if use_hybrid:
                base_query['bool']['filter'] = filter_clauses
            else:
                base_query = {
                    "bool": {
                        "must": [base_query],
                        "filter": filter_clauses
                    }
                }
    
    search_query = {
        "size": k,
        "query": base_query
    }

    try:
        response = client.search(index=index_name, body=search_query)
        
        # Extract the text chunks with their relevance scores and metadata
        results = []
        for hit in response['hits']['hits']:
            result = {
                'text': hit['_source']['text'],
                'score': hit['_score']
            }
            # Add metadata if available
            for field in ('doc_name', 'doc_type', 'source_file', 'source_type', 'table_name'):
                if field in hit['_source']:
                    result[field] = hit['_source'][field]
            
            results.append(result)
        return results
    except Exception as e:
        print(f"Error searching: {e}")
        return []

def generate_answer_with_gemini(question, context_chunks, history=None):
    """
    Use Gemini API to generate a coherent answer from retrieved chunks.

    Args:
        question: The user's question
        context_chunks: List of relevant text chunks
        history: Optional list of prior turns [{'role': 'user'|'assistant', 'content': '...'}]

    Returns:
        Generated answer string
    """
    if not gemini_model:
        return "Gemini API not configured. Please set GEMINI_API_KEY in .env file."

    # Combine context chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks[:8], 1):
        source_label = chunk.get('table_name') or chunk.get('doc_name', '')
        source_type = chunk.get('source_type', 'unknown')
        context_parts.append(f"[Context {i} | source: {source_label} ({source_type})]:\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    # Build conversation history block
    history_block = ""
    if history:
        turns = []
        for turn in history[-6:]:  # keep last 6 turns (3 rounds) to stay within token limits
            role = "User" if turn.get("role") == "user" else "Assistant"
            turns.append(f"{role}: {turn.get('content', '').strip()}")
        if turns:
            history_block = "Conversation so far:\n" + "\n".join(turns) + "\n\n"

    prompt = f"""You are a helpful assistant answering questions about laboratory data and documents.

{history_block}The context below contains relevant information retrieved for the current question.
The context may contain data from multiple related database tables or documents.
Foreign key columns (ending in _id) link records across tables. Resolved names
(ending in _name) show the human-readable value for those foreign keys.

Context:
{context}

Current question: {question}

Instructions:
- Use the conversation history to understand references to prior answers (e.g. "that program", "those labs")
- Extract and synthesize relevant information from the context
- Cross-reference data from different tables when answering
- If you find the answer in the context, provide it clearly and completely
- If information is incomplete, state what is available
- Do not add information not in the context

Answer:"""

    try:
        response = gemini_model.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating answer with Gemini: {e}"

def generate_answer_with_local_llm(question, context_chunks, history=None):
    """
    Use a local LLM to generate a coherent answer from retrieved chunks.

    Args:
        question: The user's question
        context_chunks: List of relevant text chunks
        history: Optional list of prior turns [{'role': 'user'|'assistant', 'content': '...'}]

    Returns:
        Generated answer string
    """
    load_llm()

    # Combine context chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks[:5], 1):
        source_label = chunk.get('table_name') or chunk.get('doc_name', '')
        context_parts.append(f"[Context {i} | source: {source_label}]:\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    # Build history block
    history_block = ""
    if history:
        turns = []
        for turn in history[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            turns.append(f"{role}: {turn.get('content', '').strip()}")
        if turns:
            history_block = "Conversation so far:\n" + "\n".join(turns) + "\n\n"

    prompt = f"""<|system|>
You are a precise technical assistant. Answer the question using ONLY information from the provided context and conversation history. Do not add any information not present in the context. If the context doesn't contain enough information, say "The provided context does not contain sufficient information to answer this question."</|system|>
<|user|>
{history_block}Context:
{context}

Current question: {question}

Answer based strictly on the context above:</|user|>
<|assistant|>"""

    # Tokenize and generate with more conservative parameters
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=MAX_ANSWER_LENGTH,
        temperature=0.3,  # Lower temperature for less creativity/hallucination
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    
    # Decode and extract answer
    full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|assistant|>" in full_response:
        answer = full_response.split("<|assistant|>")[-1].strip()
    else:
        answer = full_response[len(prompt):].strip()
    
    return answer

def answer_question(question, k=10, use_llm=True, doc_filter=None, history=None):
    """
    Answer a question based on indexed content.

    Args:
        question: The question to answer
        k: Number of relevant chunks to retrieve (default: 10)
        use_llm: Whether to use LLM for answer generation (default: True)
        doc_filter: Optional dict to filter by document
        history: Optional list of prior turns [{'role': 'user'|'assistant', 'content': '...'}]
                 Pass the full conversation so the LLM can resolve follow-up references.

    Returns:
        Dictionary with 'answer' and 'sources'
    """
    # Build a search query that incorporates recent history context for better retrieval
    search_query = question
    if history:
        # Prepend the last user question to improve kNN retrieval for follow-ups
        last_user = next(
            (t['content'] for t in reversed(history) if t.get('role') == 'user'), None
        )
        if last_user and last_user.strip() != question.strip():
            search_query = f"{last_user} {question}"

    effective_filter = doc_filter or infer_doc_filter_from_question(question)
    results = search_pdf(search_query, k, doc_filter=effective_filter)
    if not results and effective_filter:
        results = search_pdf(search_query, k, doc_filter=None)

    # Drop chunks below the relevance threshold
    results = [r for r in results if r['score'] >= MIN_RELEVANCE_SCORE]

    if not results:
        return {
            'answer': 'I could not find relevant information in the documents to answer that question.',
            'sources': []
        }

    if use_llm:
        if LLM_PROVIDER == 'gemini':
            answer = generate_answer_with_gemini(question, results[:8], history=history)
        else:
            answer = generate_answer_with_local_llm(question, results[:5], history=history)
    else:
        answer = "\n\n".join([f"[Relevance: {r['score']:.2f}]\n{r['text']}" for r in results])

    return {
        'answer': answer,
        'sources': results,
        'doc_filter': effective_filter
    }

if __name__ == "__main__":
    # Can be run from command line: python query.py "your question here"
    # Add --raw flag to see raw chunks without LLM processing
    # Add --debug to see all retrieved chunks
    # Add --doc "HumaCount 5D" to filter by document name
    # Add --type "Service Manual" to filter by document type
    use_llm = True
    debug = False
    doc_filter = {}
    args = sys.argv[1:]
    
    if "--raw" in args:
        use_llm = False
        args.remove("--raw")
    
    if "--debug" in args:
        debug = True
        args.remove("--debug")
    
    # Parse document filter
    if "--doc" in args:
        idx = args.index("--doc")
        doc_filter['doc_name'] = args[idx + 1]
        args.pop(idx)
        args.pop(idx)
    
    if "--type" in args:
        idx = args.index("--type")
        doc_filter['doc_type'] = args[idx + 1]
        args.pop(idx)
        args.pop(idx)
    
    if len(args) > 0:
        question = " ".join(args)
    else:
        question = input("Enter your question: ")
    
    print(f"\nSearching for: {question}")
    if doc_filter:
        print(f"Filtering by: {doc_filter}")
    print("\n" + "=" * 80)
    
    result = answer_question(question, use_llm=use_llm, doc_filter=doc_filter)
    
    if result.get('doc_filter') and not doc_filter:
        print(f"Auto-filtering by: {result['doc_filter']}")

    if result.get('doc_filter'):
        print(f"Auto-filtering by: {result['doc_filter']}")

    print(f"\nAnswer:\n{result['answer']}")
    print("\n" + "=" * 80)
    print(f"\nFound {len(result['sources'])} relevant chunks from the PDF.")
    
    # Show the source chunks with metadata
    num_chunks_to_show = len(result['sources']) if debug else min(5, len(result['sources']))
    if use_llm or debug:
        print(f"\nTop {num_chunks_to_show} retrieved context chunks:")
        for i, source in enumerate(result['sources'][:num_chunks_to_show], 1):
            doc_info = f"{source.get('doc_name', 'Unknown')} - {source.get('doc_type', 'Unknown')}"
            print(f"\n[Chunk {i}] (Relevance: {source['score']:.3f}) [{doc_info}]")
            preview = source['text'][:300] + "..." if len(source['text']) > 300 else source['text']
            print(preview)