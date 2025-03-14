import os
import json
import time
from openai import OpenAI

class DeepSeekContentAnalyzer:
    def __init__(self):
        """Initialize the Content Analyzer with DeepSeek API"""
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        
        # Define the static system prompt for caching benefit
        self.system_prompt = """
        Here's the updated prompt for the content analyzer:

        ```
        You are a news analyst with extraordinary content analysis skills.
                
        Given content text, you will:
        1. Identify a SINGLE word that best describes what the content is conveying
        2. Determine the most appropriate category from this list: politics, technology, business, science, health, sports, entertainment, education
                
        For the word selection:
        - Focus on basic, primitive parts of speech: nouns (e.g., "technology"), adjectives (e.g., "innovative"), or verbs (e.g., "accelerate") 
        - Prioritize simpler, more fundamental words over complex derivatives
        - Try to vary between different parts of speech rather than always using nouns
        - Be creative to avoid duplicated words between articles
                
        Output in JSON format ONLY with two fields:
        - "word": A single word that encapsulates the main theme/topic/message
        - "category": One of the eight categories listed above
                
        Do not include any additional explanations or comments.
        ```
        """
        
        # Setup static messages for caching benefit
        self.static_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "CONTENT: Apple has announced its latest iPhone model featuring enhanced AI capabilities, improved camera quality, and longer battery life. The new model will be available for pre-order next week and is expected to hit stores by the end of the month. Analysts predict strong sales despite the higher price point compared to previous models."},
            {"role": "assistant", "content": json.dumps({
                "word": "innovation",
                "category": "technology"
            })}
        ]
    
    def analyze_content(self, content, retry_count=2):
        """Analyze content to generate key word and category"""
        # Create a truncated preview for logging (avoid logging huge texts)
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"[ContentAnalyzer] Analyzing content: {preview}")
        
        # Start with our cached prompt structure
        messages = self.static_messages.copy()
        
        # Add the content as a new user message
        # Use a simple consistent format to maximize cache hits
        messages.append({"role": "user", "content": f"CONTENT: {content}"})
        
        for attempt in range(retry_count):
            try:
                print(f"[ContentAnalyzer] Attempt {attempt+1}/{retry_count} - Making API call to DeepSeek...")
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    response_format={'type': 'json_object'},
                    max_tokens=150,  # Limit tokens since we only need a short summary, word, and category
                    temperature=0.3  # Lower temperature for more consistent, predictable outputs
                )
                
                api_call_time = time.time() - start_time
                print(f"[ContentAnalyzer] API call completed in {api_call_time:.2f}s")
                
                content_json = response.choices[0].message.content
                
                # Check if we got valid content
                if content_json and len(content_json) > 0:
                    print(f"[ContentAnalyzer] Parsing JSON response of length {len(content_json)}")
                    result = json.loads(content_json)
                    
                    # Log cache hit information if available
                    if hasattr(response.usage, "prompt_cache_hit_tokens"):
                        cache_hit = response.usage.prompt_cache_hit_tokens
                        cache_miss = response.usage.prompt_cache_miss_tokens
                        print(f"[ContentAnalyzer] Cache hit: {cache_hit}, Cache miss: {cache_miss}")
                    
                    # Validate category is one of the allowed values
                    valid_categories = [
                        "politics", "technology", "business", "science", 
                        "health", "sports", "entertainment", "education"
                    ]
                    
                    if "category" in result and result["category"].lower() not in valid_categories:
                        print(f"[ContentAnalyzer] Invalid category '{result['category']}', defaulting to 'education'")
                        result["category"] = "education"
                    
                    print(f"[ContentAnalyzer] Successfully extracted word: '{result.get('word', 'unknown')}' and category: '{result.get('category', 'unknown')}'")
                    return result
                else:
                    print(f"[ContentAnalyzer] Empty response, retrying ({attempt+1}/{retry_count})...")
                    time.sleep(1)
            except Exception as e:
                print(f"[ContentAnalyzer] Error analyzing content: {str(e)}, retrying ({attempt+1}/{retry_count})...")
                time.sleep(1)
        
        # Return a default response if all retries fail
        print("[ContentAnalyzer] All retry attempts failed, returning default response")
        default_response = {
            "word": "Error",
            "category": "education"  # Default category
        }
        return default_response