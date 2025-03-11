import os
import json
from openai import OpenAI
from word_processor import WordProcessor

class NewsProcessor:
    def __init__(self):
        """Initialize the News Processor with DeepSeek API"""
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        
        # Initialize the WordProcessor
        self.word_processor = WordProcessor()
        
        # Define the static system prompt for caching benefit
        self.system_prompt = """
        You are a news analyst with extraordinary summarization skills.
        
        Given a news article, you will:
        1. Create a concise summary of 30-50 words that captures the essential information
        2. Identify a SINGLE word that best describes what the news is conveying
        
        Output in JSON format ONLY with two fields:
        - "summary": A 30-50 word summary of the news content
        - "word": A single word that encapsulates the main theme/topic/message
        
        Do not include any additional explanations or comments.
        """
        
        # Setup static messages for caching benefit
        self.static_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "NEWS: Apple has announced its latest iPhone model featuring enhanced AI capabilities, improved camera quality, and longer battery life. The new model will be available for pre-order next week and is expected to hit stores by the end of the month. Analysts predict strong sales despite the higher price point compared to previous models."},
            {"role": "assistant", "content": json.dumps({
                "summary": "Apple's new iPhone model with enhanced AI, better camera, and longer battery life will be available for pre-order next week with in-store availability by month-end. Strong sales expected despite higher pricing.",
                "word": "Innovation"
            })}
        ]
    
    def process_news(self, news_content, retry_count=2):
        """Process news content to generate summary and key word"""
        # Create a truncated preview for logging (avoid logging huge texts)
        preview = news_content[:100] + "..." if len(news_content) > 100 else news_content
        print(f"Processing news: {preview}")
        
        # Start with our cached prompt structure
        messages = self.static_messages.copy()
        
        # Add the news content as a new user message
        # Use a simple consistent format to maximize cache hits
        messages.append({"role": "user", "content": f"NEWS: {news_content}"})
        
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    response_format={'type': 'json_object'},
                    max_tokens=150,  # Limit tokens since we only need a short summary and one word
                    temperature=0.3  # Lower temperature for more consistent, predictable outputs
                )
                
                content = response.choices[0].message.content
                
                # Check if we got valid content
                if content and len(content) > 0:
                    result = json.loads(content)
                    
                    # Log cache hit information if available
                    if hasattr(response.usage, "prompt_cache_hit_tokens"):
                        cache_hit = response.usage.prompt_cache_hit_tokens
                        cache_miss = response.usage.prompt_cache_miss_tokens
                        print(f"Cache hit: {cache_hit}, Cache miss: {cache_miss}")
                    
                    # Process the generated word with WordProcessor
                    if 'word' in result and result['word']:
                        word_analysis = self.word_processor.process_word(result['word'])
                        result['wordAnalysis'] = word_analysis
                    
                    return result
                else:
                    print(f"Empty response, retrying ({attempt+1}/{retry_count})...")
            except Exception as e:
                print(f"Error processing news: {str(e)}, retrying ({attempt+1}/{retry_count})...")
        
        # Return a default response if all retries fail
        default_word = "Error"
        default_response = {
            "summary": "Failed to generate summary for the provided news content.",
            "word": self.word_processor.process_word(default_word)
        }
        return default_response