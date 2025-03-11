from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Import your custom modules
from word_processor import WordProcessor
from example_generator import ExampleGenerator

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize word processor and example generator
word_processor = WordProcessor()
example_generator = ExampleGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/process-word', methods=['POST'])
def process_word():
    """Process a single word"""
    data = request.get_json()
    
    if not data or 'word' not in data:
        return jsonify({"error": "No word provided"}), 400
    
    word = data['word']
    result = word_processor.process_word(word)
    
    return jsonify(result)

@app.route('/process-batch', methods=['POST'])
def process_batch():
    """Process multiple words at once"""
    data = request.get_json()
    
    if not data or 'words' not in data:
        return jsonify({"error": "No words provided"}), 400
    
    words = data['words']
    results = {}
    
    for word in words:
        results[word] = word_processor.process_word(word)
    
    return jsonify(results)

@app.route('/generate-examples', methods=['POST'])
def generate_examples():
    """Generate example sentences for a word at different levels"""
    data = request.get_json()
    
    if not data or 'word' not in data:
        return jsonify({"error": "No word provided"}), 400
    
    word = data['word']
    result = example_generator.generate_examples(word)
    
    return jsonify(result)

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)