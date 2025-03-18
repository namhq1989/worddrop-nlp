from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import time
import signal
from datetime import datetime

# Import your custom modules
from word_processor import WordProcessor
from deepseek_example_generator import DeepSeekExampleGenerator
from deepseek_content_analyzer import DeepSeekContentAnalyzer
from datamuse_api import DatamuseAPI
from openrouter_content_analyzer import OpenRouterContentAnalyzer
from openrouter_example_generator import OpenRouterExampleGenerator
from openrouter_usage_tracker import OpenRouterUsageTracker

# Load environment variables
load_dotenv()

# Get AI provider to use
AI_PROVIDER = os.environ.get('AI_PROVIDER', 'deepseek').lower()
print(f"Using AI provider: {AI_PROVIDER}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize processors
word_processor = WordProcessor()
datamuse_api = DatamuseAPI()

# Initialize OpenRouter usage tracker if needed
openrouter_tracker = None
if AI_PROVIDER == 'openrouter':
    openrouter_tracker = OpenRouterUsageTracker()

# Initialize AI-based processors based on configuration
if AI_PROVIDER == 'openrouter':
    content_analyzer = OpenRouterContentAnalyzer(openrouter_tracker)
    example_generator = OpenRouterExampleGenerator(openrouter_tracker)
    print("Initialized OpenRouter AI processors")
else:  # Default to DeepSeek
    content_analyzer = DeepSeekContentAnalyzer()
    example_generator = DeepSeekExampleGenerator()
    print("Initialized DeepSeek AI processors")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok", 
        "provider": AI_PROVIDER,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/openrouter-usage', methods=['GET'])
def usage_stats():
    """Get OpenRouter usage statistics"""
    if AI_PROVIDER != 'openrouter' or not openrouter_tracker:
        return jsonify({
            "error": "Usage tracking is only available for the OpenRouter provider",
            "current_provider": AI_PROVIDER
        }), 400
    
    stats = openrouter_tracker.get_usage_stats()
    return jsonify(stats)

@app.route('/extract-word', methods=['POST'])
def extract_word():
    """
    Extract a key descriptive word from content using the configured AI provider.
    Takes content as input and returns key descriptive word and category.
    """
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-----------------------------")
    print(f"[LOG {current_time}] Starting extract-word endpoint (provider: {AI_PROVIDER})")
    
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
    
    # Extract key word and determine category using configured provider
    print(f"[LOG] Extracting key word and category using {AI_PROVIDER}...")
    start_time = time.time()
    try:
        analysis_result = content_analyzer.analyze_content(news_content)
        
        # Ensure word is in lowercase
        if 'word' in analysis_result:
            analysis_result['word'] = analysis_result['word'].lower()

             # Process the extracted word to check for base forms
            print(f"[LOG] Processing extracted word: '{analysis_result['word']}'")
            extracted_word = analysis_result['word']
            
            # Use WordProcessor to analyze the word and find base forms
            word_info = word_processor.process_word(extracted_word)
            
            # Check if the word has a different base form as a verb or noun
            original_word = extracted_word
            if 'verb' in word_info and 'base' in word_info['verb'] and word_info['verb']['base'] != extracted_word:
                analysis_result['word'] = word_info['verb']['base']
                print(f"[LOG] Updated word from '{original_word}' to verb base form '{analysis_result['word']}'")
            elif 'noun' in word_info and 'base' in word_info['noun'] and word_info['noun']['base'] != extracted_word:
                analysis_result['word'] = word_info['noun']['base']
                print(f"[LOG] Updated word from '{original_word}' to noun base form '{analysis_result['word']}'")
            
        # Add provider to the response
        analysis_result['provider'] = AI_PROVIDER
            
        print(f"[LOG] Analysis completed in {time.time() - start_time:.2f}s - Word: '{analysis_result['word']}', Category: '{analysis_result['category']}'")
        return jsonify(analysis_result)
    except Exception as e:
        print(f"[ERROR] Error in content analysis: {str(e)}")
        return jsonify({"error": f"Failed to analyze content: {str(e)}"}), 500

@app.route('/analyze-word', methods=['POST'])
def analyze_word():
    """
    Analyze a word:
    1. Validate the word exists in dictionary
    2. Analyze the word's linguistic properties
    3. Generate examples for the word using the configured AI provider
    4. Return comprehensive word information
    """
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-----------------------------")
    print(f"[LOG {current_time}] Starting analyze-word endpoint (provider: {AI_PROVIDER})")
    
    data = request.get_json()
    
    if not data or 'word' not in data:
        print("[ERROR] No word provided in request")
        return jsonify({"error": "No word provided"}), 400
    
    # Get the word to analyze
    word = data['word'].strip()
    
    # Get category if provided, otherwise default to 'education'
    category = data.get('category', 'education')
    
    if not word:
        print("[ERROR] Empty word provided")
        return jsonify({"error": "Empty word provided"}), 400
    
    print(f"[LOG] Received word: '{word}', category: '{category}'")
    
    # Define timeout for operations
    TIMEOUT = 60
    
    # Setup timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    # STEP 1: Validate the word exists in dictionary using Datamuse API
    print(f"[LOG] Step 1: Validating word '{word}' in dictionary...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        # Check if the word exists in the dictionary using Datamuse API
        dictionary_data = datamuse_api.search_term(word)
        
        # Cancel the alarm
        signal.alarm(0)
        
        # If no definitions were found, the word might not exist in the dictionary
        if not dictionary_data or not dictionary_data.get('definitions'):
            print(f"[ERROR] Word '{word}' not found in dictionary")
            return jsonify({
                "error": f"The word '{word}' was not found in our dictionary. Please try a different word."
            }), 404
        
        print(f"[LOG] Step 1 completed in {time.time() - start_time:.2f}s - Word verified in dictionary")
    except TimeoutError:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Dictionary validation timed out after {TIMEOUT} seconds")
        return jsonify({"error": "Dictionary validation timed out. Please try again later."}), 504
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Exception in dictionary validation: {str(e)}")
        return jsonify({"error": f"Failed to validate word: {str(e)}"}), 500
    
    # STEP 2: Analyze word with timeout protection
    print(f"[LOG] Step 2: Analyzing word '{word}'...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        word_analysis = word_processor.process_word(word)
        
        # Cancel the alarm
        signal.alarm(0)
        print(f"[LOG] Step 2 completed in {time.time() - start_time:.2f}s")
    except TimeoutError:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Word analysis timed out after {TIMEOUT} seconds")
        return jsonify({"error": "Word analysis timed out. Please try again later."}), 504
    except Exception as e:
        signal.alarm(0)  # Ensure alarm is canceled
        print(f"[ERROR] Exception in word analysis: {str(e)}")
        return jsonify({"error": f"Failed to analyze word: {str(e)}"}), 500
    
    # STEP 3: Generate examples with timeout protection using configured AI provider
    print(f"[LOG] Step 3: Generating examples for '{word}' using {AI_PROVIDER}...")
    start_time = time.time()
    try:
        # Set up timeout for this operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)
        
        examples_data = example_generator.generate_examples(word)
        
        # Cancel the alarm
        signal.alarm(0)
        print(f"[LOG] Step 3 completed in {time.time() - start_time:.2f}s")
        
        # Validate examples data structure
        if not examples_data or not isinstance(examples_data, dict) or 'examples' not in examples_data:
            print("[ERROR] Invalid example generation result")
            return jsonify({"error": "Failed to generate examples. Invalid result structure."}), 500
        
        # Add provider to examples data
        examples_data['provider'] = AI_PROVIDER
        
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

    # Update word to base form if it's different
    original_word = word
    if 'verb' in word_analysis and word_analysis['verb'].get('base') and word_analysis['verb']['base'] != word:
        word = word_analysis['verb']['base']
        word_analysis['word'] = word
        print(f"[LOG] Updated word from '{original_word}' to base verb form '{word}'")
    elif 'noun' in word_analysis and word_analysis['noun'].get('base') and word_analysis['noun']['base'] != word:
        word = word_analysis['noun']['base']
        word_analysis['word'] = word
        print(f"[LOG] Updated word from '{original_word}' to base noun form '{word}'")
    
    print("[LOG] All processing completed, preparing response")
    
    # Combine all results
    result = {
        "category": category,
        "word": word_analysis,
        "examples": examples,
        "provider": AI_PROVIDER
    }
    
    print("[LOG] Response ready")
    return jsonify(result)