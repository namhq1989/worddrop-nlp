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

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize processors
word_processor = WordProcessor()
example_generator = ExampleGenerator()
content_analyzer = ContentAnalyzer()

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
    
    # Setup default values in case of failures
    default_word_analysis = {
        "word": "Unknown",
        "ipa": None,
        "pos": ["noun"]
    }
    
    default_examples = {
        "beginner": {"example": "This is a basic example.", "word": "Unknown"},
        "intermediate": {"example": "This represents an intermediate level example.", "word": "Unknown"},
        "advanced": {"example": "The intricate nuances of this advanced example demonstrate comprehensive language mastery.", "word": "Unknown"}
    }
    
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
    
    # STEP 2: Analyze word with timeout protection
    print(f"[LOG] Step 2: Analyzing word '{extracted_word}'...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        word_analysis = word_processor.process_word(extracted_word)
        
        # Cancel the alarm
        signal.alarm(0)
        print(f"[LOG] Step 2 completed in {time.time() - start_time:.2f}s")
    except TimeoutError:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Word analysis timed out after {TIMEOUT} seconds")
        word_analysis = default_word_analysis
        word_analysis["word"] = extracted_word
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Exception in word analysis: {str(e)}")
        word_analysis = default_word_analysis
        word_analysis["word"] = extracted_word
    
    # STEP 3: Generate examples with timeout protection
    print(f"[LOG] Step 3: Generating examples for '{extracted_word}'...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        examples_data = example_generator.generate_examples(extracted_word)
        
        # Cancel the alarm
        signal.alarm(0)
        print(f"[LOG] Step 3 completed in {time.time() - start_time:.2f}s")
        
        # Validate examples data structure
        if not examples_data or not isinstance(examples_data, dict) or 'examples' not in examples_data:
            print("[ERROR] Invalid example generation result")
            examples = default_examples
        else:
            examples = examples_data.get('examples', default_examples)
    except TimeoutError:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Example generation timed out after {TIMEOUT} seconds")
        examples = default_examples
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Exception in example generation: {str(e)}")
        examples = default_examples
    
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