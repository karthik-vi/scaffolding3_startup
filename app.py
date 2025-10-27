"""
app.py
Flask application template for the warm-up assignment

Students need to implement the API endpoints as specified in the assignment.
"""

from flask import Flask, request, jsonify, render_template
from starter_preprocess import TextPreprocessor
import traceback

app = Flask(__name__)
preprocessor = TextPreprocessor()


@app.route('/')
def home():
    """Render a simple HTML form for URL input"""
    return render_template('index.html')


@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Text preprocessing service is running"
    })


@app.route('/api/clean', methods=['POST'])
def clean_text():
    """
    TODO: Implement this endpoint for Part 3
    ... (docstring) ...
    """
    try:
        # 1. Get JSON data from request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"success": False, "error": "Missing 'url' in JSON payload"}), 400

        # 2. Extract URL
        url = data['url']

        # 3. Fetch text (fetch_from_url will validate .txt and raise errors)
        raw_text = preprocessor.fetch_from_url(url)

        # 4. Clean and normalize the text
        # Assuming these methods exist in your starter_preprocess.py
        cleaned_text = preprocessor.clean_gutenberg_text(raw_text)
        normalized_text = preprocessor.normalize_text(cleaned_text)

        # 5. Get statistics
        statistics = preprocessor.get_text_statistics(normalized_text)

        # 6. Create summary
        summary = preprocessor.create_summary(normalized_text, num_sentences=3)

        # 7. Return the full JSON response
        return jsonify({
            "success": True,
            # The PDF asks for 'cleaned_text', so we'll return the normalized one
            "cleaned_text": normalized_text,
            "statistics": statistics,
            "summary": summary
        })

    except Exception as e:
        # This will catch errors from fetch_from_url or other steps
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400  # 400 is better for client-side errors like bad URLs


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """
    TODO: Implement this endpoint for Part 3
    ... (docstring) ...
    """
    try:
        # 1. Get JSON data from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "Missing 'text' in JSON payload"}), 400

        # 2. Extract text
        text = data['text']

        # 3. Get statistics
        statistics = preprocessor.get_text_statistics(text)

        # 4. Return JSON response
        return jsonify({
            "success": True,
            "statistics": statistics
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

# Error handlers


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    print("🚀 Starting Text Preprocessing Web Service...")
    print("📖 Available endpoints:")
    print("   GET  /           - Web interface")
    print("   GET  /health     - Health check")
    print("   POST /api/clean  - Clean text from URL")
    print("   POST /api/analyze - Analyze raw text")
    print()
    print("🌐 Open your browser to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")

    app.run(debug=True, port=5000, host='0.0.0.0')
