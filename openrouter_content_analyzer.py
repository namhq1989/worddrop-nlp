from openrouter_handler import OpenRouterHandler

class OpenRouterContentAnalyzer(OpenRouterHandler):
    """Content analyzer using OpenRouter API"""
    
    def __init__(self, usage_tracker):
        """Initialize the Content Analyzer"""
        super().__init__(usage_tracker)
        
        # Define the system prompt
        self.system_prompt = """
        You are a news analyst with extraordinary content analysis skills.
                
        Given content text, you will:
        1. Identify a SINGLE word that best describes what the content is conveying
        2. Determine the most appropriate category from this list: politics, technology, business, science, health, sports, entertainment, education
                
        For the word selection:
        - Focus on basic, primitive parts of speech: nouns (e.g., "technology"), adjectives (e.g., "innovative"), or verbs (e.g., "accelerate") 
        - Prioritize simpler, more fundamental words over complex derivatives
        - Try to vary between different parts of speech rather than always using nouns
        - Be creative to avoid duplicated words between articles
                
        Output in JSON format with exactly two fields:
        - "word": A single word that encapsulates the main theme/topic/message
        - "category": One of the eight categories listed above
        """
    
    def analyze_content(self, content, retry_count=2):
        """Analyze content to generate key word and category using OpenRouter API"""
        # Create a truncated preview for logging
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"[OpenRouterAnalyzer] Analyzing content: {preview}")
        
        # Get the best model to use for this request
        model = self.usage_tracker.get_next_available_model()
        if not model:
            raise ValueError("No OpenRouter models available. All models may have reached their daily limits.")
        
        # Format messages based on the model type
        messages = self._format_messages_for_model(model, f"CONTENT: {content}", self.system_prompt)
        
        # Call the model
        result = self.call_model(model, messages, max_tokens=150, retry_count=retry_count)
        
        if result:
            # Validate category is one of the allowed values
            valid_categories = [
                "politics", "technology", "business", "science", 
                "health", "sports", "entertainment", "education"
            ]
            
            if "category" in result and result["category"].lower() not in valid_categories:
                print(f"[OpenRouterAnalyzer] Invalid category '{result['category']}', defaulting to 'education'")
                result["category"] = "education"
            
            # Ensure word is lowercase
            if "word" in result:
                result["word"] = result["word"].lower()
                
            # Add provider to the response
            result["provider"] = "openrouter"
                
            print(f"[OpenRouterAnalyzer] Successfully extracted word: '{result.get('word', 'unknown')}' and category: '{result.get('category', 'unknown')}'")
            return result
        
        # Default response if all attempts fail
        print("[OpenRouterAnalyzer] All attempts failed, returning default response")
        default_response = {
            "word": "error",
            "category": "error",
            "model": "fallback",
            "provider": "openrouter"
        }
        return default_response