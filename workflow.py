"""
Core workflow module for web scraping, NLP ranking, and PDF generation (local storage).
"""

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
import re
from datetime import datetime
import os
import uuid

# Path to store generated PDFs
PDF_STORAGE_DIR = 'generated_pdfs'

# Initialize the Sentence Transformer model (will download on first run)
model = SentenceTransformer('all-MiniLM-L6-v2')


def fetch_and_parse_content(url):
    """
    Stage 1 & 2: Fetch HTML content and parse it to extract essay-like content.
    Returns a list of dictionaries with 'title' and 'content' keys.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Generalized content extraction strategy
        essays = []
        
        # Strategy 1: Look for article tags
        articles = soup.find_all('article')
        if articles:
            for idx, article in enumerate(articles[:10]):  # Limit to top 10
                title = article.find(['h1', 'h2', 'h3'])
                title_text = title.get_text(strip=True) if title else f"Article {idx+1}"
                
                paragraphs = article.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                
                if len(content) > 200:  # Minimum content threshold
                    essays.append({
                        'title': title_text,
                        'content': content
                    })
        
        # Strategy 2: Look for main content div/section
        if not essays:
            main_content = soup.find(['main', 'article']) or soup.find('div', class_=re.compile(r'(content|post|entry|article)', re.I))
            
            if main_content:
                # Extract title
                title_elem = main_content.find(['h1', 'h2'])
                page_title = title_elem.get_text(strip=True) if title_elem else soup.title.string if soup.title else "Web Content"
                
                # Extract all paragraphs
                paragraphs = main_content.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                
                if len(content) > 200:
                    essays.append({
                        'title': page_title,
                        'content': content
                    })
        
        # Strategy 3: Fallback - extract all paragraphs from body
        if not essays:
            page_title = soup.title.string if soup.title else "Web Content"
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            
            if len(content) > 200:
                essays.append({
                    'title': page_title,
                    'content': content
                })
        
        return essays if essays else [{'title': 'No Content Found', 'content': 'Unable to extract meaningful content from this URL.'}]
    
    except Exception as e:
        raise Exception(f"Error fetching/parsing content: {str(e)}")


def rank_content_by_relevance(essays, user_query):
    """
    Use Sentence Transformers to rank essays by semantic similarity to the user query.
    Returns the top-ranked essay.
    """
    if not essays:
        return None
    
    try:
        # Encode the user query
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        
        # Encode all essay contents (use truncated version for efficiency)
        essay_texts = [essay['content'][:1000] for essay in essays]
        essay_embeddings = model.encode(essay_texts, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.cos_sim(query_embedding, essay_embeddings)[0]
        
        # Find the highest scoring essay
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        
        print(f"Best match: '{essays[best_idx]['title']}' with score {best_score:.4f}")
        
        return essays[best_idx]
    
    except Exception as e:
        print(f"Error in NLP ranking: {str(e)}. Returning first essay.")
        return essays[0]


def generate_pdf(essay_data, source_url, filename):
    """
    Stage 3: Generate a professional PDF and save it locally.
    Returns the file path.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.multi_cell(0, 10, essay_data['title'], align='C')
        pdf.ln(5)
        
        # Metadata
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, f"Source: {source_url}", ln=True)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        
        # Query information (if provided)
        if hasattr(essay_data, 'query_used') and essay_data.get('query_used'):
            pdf.cell(0, 10, f"Research Query: {essay_data['query_used']}", ln=True)
            pdf.ln(5)
        
        # Content
        pdf.set_font('Arial', '', 11)
        pdf.set_text_color(0, 0, 0)
        
        # Clean and format content
        content = essay_data['content']
        # Replace multiple spaces with single space
        content = re.sub(r'\s+', ' ', content)
        
        # Add word wrap and proper formatting
        pdf.multi_cell(0, 7, content)
        
        # Ensure storage directory exists
        os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
        
        # Full file path
        file_path = os.path.join(PDF_STORAGE_DIR, filename)
        
        # Save PDF locally
        pdf.output(file_path)
        
        print(f"✅ PDF saved locally: {file_path}")
        print(f"📊 File size: {os.path.getsize(file_path)} bytes")
        
        return file_path
    
    except Exception as e:
        raise Exception(f"Error generating PDF: {str(e)}")


def generate_filename(essay_title, query):
    """
    Generate a clean filename from essay title and query.
    """
    # Clean the title for filename
    clean_title = re.sub(r'[^\w\s-]', '', essay_title)
    clean_title = re.sub(r'[-\s]+', '_', clean_title)
    clean_title = clean_title.strip('_')
    
    # Clean the query for filename
    clean_query = re.sub(r'[^\w\s-]', '', query)
    clean_query = re.sub(r'[-\s]+', '_', clean_query)
    clean_query = clean_query.strip('_')
    
    # Truncate if too long
    clean_title = clean_title[:50]
    clean_query = clean_query[:30]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if clean_title and clean_query:
        filename = f"NotebookLM_{clean_title}_{clean_query}_{timestamp}.pdf"
    elif clean_title:
        filename = f"NotebookLM_{clean_title}_{timestamp}.pdf"
    else:
        filename = f"NotebookLM_Content_{timestamp}.pdf"
    
    return filename


def process_workflow(url, user_query):
    """
    Main workflow orchestrator.
    Returns the local file path of the generated PDF.
    """
    print(f"Starting workflow for URL: {url}")
    print(f"User query: {user_query}")
    
    # Stage 1 & 2: Fetch and parse content
    essays = fetch_and_parse_content(url)
    print(f"Found {len(essays)} content sections")
    
    # Rank by relevance
    best_essay = rank_content_by_relevance(essays, user_query)
    
    if not best_essay:
        raise Exception("No relevant content found")
    
    # Add query to essay data for PDF generation
    best_essay['query_used'] = user_query
    
    # Generate filename
    filename = generate_filename(best_essay['title'], user_query)
    
    # Stage 3: Generate and save PDF locally
    print("Generating PDF...")
    local_file_path = generate_pdf(best_essay, url, filename)
    
    print("✅ Workflow completed successfully!")
    return local_file_path