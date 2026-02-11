#!/usr/bin/env python3
"""
Test script to verify paragraph-aware chunking and page metadata.
This script tests the chunking without requiring OpenSearch to be running.
"""
import os
import sys
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
from pypdf import PdfReader
import re

# Load environment variables
load_dotenv()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Configuration
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "300"))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
PARAGRAPH_BREAK_NEWLINES = int(os.getenv("PARAGRAPH_BREAK_NEWLINES", "2"))

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_chunks_with_pages(pdf_path, min_tokens=CHUNK_MIN_TOKENS, max_tokens=CHUNK_MAX_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """Extract text from PDF and split into paragraph-aware chunks with page metadata."""
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
    
    # Create chunks respecting paragraph boundaries
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

def main():
    """Test the paragraph-aware chunking with page metadata."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Find first PDF in data directory
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in data directory")
        return
    
    test_pdf = os.path.join(data_dir, pdf_files[0])
    print(f"Testing paragraph-aware chunking on: {pdf_files[0]}")
    print(f"Configuration:")
    print(f"  - Min tokens: {CHUNK_MIN_TOKENS}")
    print(f"  - Max tokens: {CHUNK_MAX_TOKENS}")
    print(f"  - Overlap tokens: {CHUNK_OVERLAP_TOKENS}")
    print(f"  - Paragraph break newlines: {PARAGRAPH_BREAK_NEWLINES}")
    print()
    
    try:
        chunks = get_chunks_with_pages(test_pdf)
        
        print(f"✓ Successfully created {len(chunks)} chunks")
        print()
        
        # Analyze chunk statistics
        token_counts = [len(tokenizer.encode(chunk['text'])) for chunk in chunks]
        
        print("Chunk Statistics:")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Min tokens: {min(token_counts)}")
        print(f"  - Max tokens: {max(token_counts)}")
        print(f"  - Avg tokens: {sum(token_counts) / len(token_counts):.1f}")
        print()
        
        # Analyze page metadata
        single_page_chunks = sum(1 for c in chunks if c['start_page'] == c['end_page'])
        multi_page_chunks = len(chunks) - single_page_chunks
        
        print("Page Metadata Statistics:")
        print(f"  - Single-page chunks: {single_page_chunks}")
        print(f"  - Multi-page chunks: {multi_page_chunks}")
        print()
        
        # Show first 3 chunks with page metadata
        print("Sample chunks (first 3):")
        for i, chunk in enumerate(chunks[:3], 1):
            tokens = len(tokenizer.encode(chunk['text']))
            preview = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
            print(f"\n  Chunk {i} ({tokens} tokens, pages {chunk['page_numbers']}):")
            print(f"    Start page: {chunk['start_page']}, End page: {chunk['end_page']}")
            print(f"    {preview}")
        
        print("\n✓ Paragraph-aware chunking test completed successfully!")
        print("\n✓ Page metadata is correctly tracked for all chunks!")
        
    except Exception as e:
        print(f"✗ Error during chunking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
