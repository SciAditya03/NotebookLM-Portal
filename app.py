"""
Flask application entry point for Website to NotebookLM automation.
"""

from flask import Flask, render_template, request, jsonify, send_file
from workflow import process_workflow
import os
import uuid

app = Flask(__name__)

# Directory to store generated PDFs
PDF_STORAGE_DIR = 'generated_pdfs'

# Create PDF storage directory if it doesn't exist
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)


@app.route('/')
def index():
    """
    Render the main form page.
    """
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """
    Handle the form submission, process the workflow, and return results.
    """
    try:
        # Get form data
        url = request.form.get('url', '').strip()
        query = request.form.get('query', '').strip()
        
        # Validate inputs
        if not url:
            return render_template('result.html', 
                                 error="Please provide a valid URL")
        
        if not query:
            return render_template('result.html', 
                                 error="Please provide a research query")
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Process the workflow - returns local file path
        local_file_path = process_workflow(url, query)
        
        # Extract filename for display
        filename = os.path.basename(local_file_path)
        
        return render_template('result.html', 
                             success=True,
                             filename=filename,
                             file_path=local_file_path)
    
    except Exception as e:
        print(f"Error in /process route: {str(e)}")
        return render_template('result.html', 
                             error=f"Processing failed: {str(e)}")


@app.route('/download/<filename>')
def download_pdf(filename):
    """
    Download endpoint for generated PDFs.
    """
    try:
        file_path = os.path.join(PDF_STORAGE_DIR, filename)
        
        # Security check: ensure file is within the PDF storage directory
        if not os.path.exists(file_path) or not os.path.commonpath([PDF_STORAGE_DIR, file_path]) == PDF_STORAGE_DIR:
            return render_template('result.html', 
                                 error="File not found or invalid path")
        
        return send_file(file_path, 
                        as_attachment=True, 
                        download_name=filename,
                        mimetype='application/pdf')
    
    except Exception as e:
        return render_template('result.html', 
                             error=f"Download failed: {str(e)}")


@app.route('/view/<filename>')
def view_pdf(filename):
    """
    View PDF in browser.
    """
    try:
        file_path = os.path.join(PDF_STORAGE_DIR, filename)
        
        # Security check
        if not os.path.exists(file_path) or not os.path.commonpath([PDF_STORAGE_DIR, file_path]) == PDF_STORAGE_DIR:
            return render_template('result.html', 
                                 error="File not found or invalid path")
        
        return send_file(file_path, 
                        mimetype='application/pdf')
    
    except Exception as e:
        return render_template('result.html', 
                             error=f"View failed: {str(e)}")


@app.route('/list-pdfs')
def list_pdfs():
    """
    List all generated PDFs (for debugging/management).
    """
    try:
        files = []
        for filename in os.listdir(PDF_STORAGE_DIR):
            if filename.endswith('.pdf'):
                file_path = os.path.join(PDF_STORAGE_DIR, filename)
                file_size = os.path.getsize(file_path)
                files.append({
                    'name': filename,
                    'size': f"{file_size/1024:.1f} KB",
                    'download_url': f'/download/{filename}',
                    'view_url': f'/view/{filename}'
                })
        
        return jsonify({'pdfs': files})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """
    Health check endpoint for deployment monitoring.
    """
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # Use port 8080 for Cloud Run compatibility
    port = int(os.environ.get('PORT', 8080))
    
    print(f"Starting Flask server on port {port}...")
    print(f"PDFs will be stored in: {os.path.abspath(PDF_STORAGE_DIR)}")
    app.run(host='0.0.0.0', port=port, debug=True)