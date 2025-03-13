from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import time
import signal

# Import your custom modules
from word_processor import WordProcessor
from example_generator import ExampleGenerator
from content_analyzer import ContentAnalyzer
from datamuse_api import DatamuseAPI  # Import the Datamuse API module

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize processors
word_processor = WordProcessor()
example_generator = ExampleGenerator()
content_analyzer = ContentAnalyzer()
datamuse_api = DatamuseAPI()  # Initialize the Datamuse API client

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/analyze-content', methods=['POST'])
def analyze_content():
    """
    Process news content:
    1. Extract a key descriptive word
    2. Determine content category
    3. Analyze the word's linguistic properties
    4. Generate examples for the word
    5. Get additional information from Datamuse API
    """
    
    print("-----------------------------")
    print("[LOG] Starting analyze-content endpoint")
    
    data = request.get_json()
    
    if not data or 'content' not in data:
        print("[ERROR] No content provided in request")
        return jsonify({"error": "No content provided"}), 400
    
    news_content = data['content']
    print(f"[LOG] Received content of length: {len(news_content)} characters")
    
    # Check if content is too short
    if len(news_content.split()) < 20:
        print("[ERROR] Content too short")
        return jsonify({"error": "Content too short. Please provide at least 20 words."}), 400
    
    # Define timeout for operations
    TIMEOUT = 15  # seconds
    
    # Setup timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    # STEP 1: Extract key word and determine category
    print("[LOG] Step 1: Extracting key word and category...")
    start_time = time.time()
    try:
        analysis_result = content_analyzer.analyze_content(news_content)
        print(f"[LOG] Step 1 completed in {time.time() - start_time:.2f}s - Word: '{analysis_result['word']}', Category: '{analysis_result['category']}'")
    except Exception as e:
        print(f"[ERROR] Error in content analysis: {str(e)}")
        return jsonify({"error": f"Failed to analyze content: {str(e)}"}), 500
    
    # Get the extracted word
    extracted_word = analysis_result['word']
    
    # STEP 2: Validate the word exists in dictionary using Datamuse API
    print(f"[LOG] Step 2: Validating word '{extracted_word}' in dictionary...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        # Check if the word exists in the dictionary using Datamuse API
        dictionary_data = datamuse_api.search_term(extracted_word)
        
        # Cancel the alarm
        signal.alarm(0)
        
        # If no definitions were found, the word might not exist in the dictionary
        if not dictionary_data or not dictionary_data.get('definitions'):
            print(f"[ERROR] Word '{extracted_word}' not found in dictionary")
            return jsonify({
                "error": f"The word '{extracted_word}' was not found in our dictionary. Please try a different text."
            }), 404
        
        print(f"[LOG] Step 2 completed in {time.time() - start_time:.2f}s - Word verified in dictionary")
    except TimeoutError:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Dictionary validation timed out after {TIMEOUT} seconds")
        return jsonify({"error": "Dictionary validation timed out. Please try again later."}), 504
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Exception in dictionary validation: {str(e)}")
        return jsonify({"error": f"Failed to validate word: {str(e)}"}), 500
    
    # STEP 3: Analyze word with timeout protection
    print(f"[LOG] Step 3: Analyzing word '{extracted_word}'...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        word_analysis = word_processor.process_word(extracted_word)
        
        # Cancel the alarm
        signal.alarm(0)
        print(f"[LOG] Step 3 completed in {time.time() - start_time:.2f}s")
    except TimeoutError:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Word analysis timed out after {TIMEOUT} seconds")
        return jsonify({"error": "Word analysis timed out. Please try again later."}), 504
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Exception in word analysis: {str(e)}")
        return jsonify({"error": f"Failed to analyze word: {str(e)}"}), 500
    
    # STEP 4: Generate examples with timeout protection
    print(f"[LOG] Step 4: Generating examples for '{extracted_word}'...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        examples_data = example_generator.generate_examples(extracted_word)
        
        # Cancel the alarm
        signal.alarm(0)
        print(f"[LOG] Step 4 completed in {time.time() - start_time:.2f}s")
        
        # Validate examples data structure
        if not examples_data or not isinstance(examples_data, dict) or 'examples' not in examples_data:
            print("[ERROR] Invalid example generation result")
            return jsonify({"error": "Failed to generate examples. Invalid result structure."}), 500
        
        examples = examples_data.get('examples')
    except TimeoutError:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Example generation timed out after {TIMEOUT} seconds")
        return jsonify({"error": "Example generation timed out. Please try again later."}), 504
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Exception in example generation: {str(e)}")
        return jsonify({"error": f"Failed to generate examples: {str(e)}"}), 500
    
    # Get word level from frequency using the WordProcessor method
    word_level = word_processor.classify_word_level(dictionary_data.get('frequency', 0))
    
    # Enhance word_analysis with dictionary data
    if 'ipa' not in word_analysis or not word_analysis['ipa']:
        word_analysis['ipa'] = dictionary_data.get('ipa', '')
    
    # Add level to word_analysis
    word_analysis['level'] = word_level
    
    # Add definitions from dictionary data
    word_analysis['definitions'] = dictionary_data.get('definitions', [])
    
    print("[LOG] All processing completed, preparing response")
    
    # Combine all results
    result = {
        "category": analysis_result.get('category', 'education'),
        "word": word_analysis,
        "examples": examples
    }
    
    print("[LOG] Response ready")
    return jsonify(result)

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)