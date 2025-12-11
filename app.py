"""
Flask application entry point for Website to NotebookLM automation.
This version starts a background thread to load heavy ML libraries/models
so the web worker comes up fast and Render/Gunicorn health checks succeed.
"""

import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Directory to store generated PDFs
PDF_STORAGE_DIR = 'generated_pdfs'
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Globals for lazy-loaded model/workflow
model = None
model_ready = False
workflow_ready = False
process_workflow = None
_model_lock = threading.Lock()


def load_model_and_workflow():
    """
    Background loader that imports heavy ML libraries and the workflow module.
    This avoids blocking the web worker startup.
    """
    global model, model_ready, process_workflow, workflow_ready

    try:
        app.logger.info("Background loader starting: importing heavy libs...")
        # Import heavy libs inside the thread
        # Replace or adjust model name/path as appropriate for your project
        from sentence_transformers import SentenceTransformer

        # Example small model — replace with your actual model path if needed.
        # NOTE: using smaller model reduces memory pressure.
        model = SentenceTransformer('all-MiniLM-L6-v2')

        with _model_lock:
            model_ready = True
        app.logger.info("SentenceTransformer model loaded successfully.")

        # Now import the workflow module (which may depend on sentence_transformers / torch)
        # We import here so imports happen after heavy libs are available and won't block main thread.
        try:
            import workflow as wf  # local module
            # assign the function reference we need
            process_workflow = getattr(wf, "process_workflow", None)
            if process_workflow is None:
                app.logger.warning("workflow module imported but 'process_workflow' not found.")
            else:
                workflow_ready = True
                app.logger.info("workflow imported and ready.")
        except Exception as e:
            app.logger.exception("Failed to import workflow module inside background loader: %s", e)

    except Exception as e:
        app.logger.exception("Exception while loading model or workflow: %s", e)
        # keep flags False — main app will show useful message via /health and /process


# Start background thread immediately so the worker starts fast
_loader_thread = threading.Thread(target=load_model_and_workflow, daemon=True)
_loader_thread.start()


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
    This waits for workflow readiness and returns a friendly error page while loading.
    """
    try:
        # Get form data
        url = request.form.get('url', '').strip()
        query = request.form.get('query', '').strip()

        # Validate inputs
        if not url:
            return render_template('result.html', error="Please provide a valid URL")

        if not query:
            return render_template('result.html', error="Please provide a research query")

        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # If workflow isn't ready yet, return a friendly "still loading" page
        if not workflow_ready or not model_ready or process_workflow is None:
            # Optionally include an ETA or suggestions in the template
            return render_template('result.html',
                                   error="Service is still starting (model/workflow loading). Please try again in a few moments.")

        # Process the workflow - returns local file path
        local_file_path = process_workflow(url, query)

        # Extract filename for display
        filename = os.path.basename(local_file_path)

        return render_template('result.html',
                               success=True,
                               filename=filename,
                               file_path=local_file_path)

    except Exception as e:
        app.logger.exception("Error in /process route: %s", e)
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
        if not os.path.exists(file_path) or not os.path.commonpath([os.path.abspath(PDF_STORAGE_DIR), os.path.abspath(file_path)]) == os.path.abspath(PDF_STORAGE_DIR):
            return render_template('result.html', error="File not found or invalid path")

        return send_file(file_path,
                         as_attachment=True,
                         download_name=filename,
                         mimetype='application/pdf')

    except Exception as e:
        app.logger.exception("Error in /download route: %s", e)
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
        if not os.path.exists(file_path) or not os.path.commonpath([os.path.abspath(PDF_STORAGE_DIR), os.path.abspath(file_path)]) == os.path.abspath(PDF_STORAGE_DIR):
            return render_template('result.html', error="File not found or invalid path")

        return send_file(file_path,
                         mimetype='application/pdf')

    except Exception as e:
        app.logger.exception("Error in /view route: %s", e)
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
        app.logger.exception("Error in /list-pdfs route: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """
    Health check endpoint for deployment monitoring.
    """
    return jsonify({
        'status': 'healthy',
        'model_ready': model_ready,
        'workflow_ready': workflow_ready
    }), 200


# Keep this block for local dev. Under Gunicorn, Render will import the app and run workers.
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.logger.info(f"Starting Flask server on port {port}...")
    app.logger.info(f"PDFs will be stored in: {os.path.abspath(PDF_STORAGE_DIR)}")
    app.run(host='0.0.0.0', port=port, debug=True)
