import os
import json
import time
import re
import httpx
from openai import OpenAI

class OpenRouterHandler:
    """
    Base class for making OpenRouter API calls with proper JSON formatting
    """
    
    def __init__(self, usage_tracker):
        """Initialize with OpenRouter API"""
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # Set proper timeouts for the httpx client
        # Note: OpenAI client uses httpx internally
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            # Use a strict timeout that won't hang for too long
            timeout=httpx.Timeout(60.0, connect=5.0, read=25.0, write=5.0),
        )
        
        # Store the usage tracker
        self.usage_tracker = usage_tracker
        
        # Define site information for OpenRouter headers
        self.site_url = os.environ.get('SITE_URL', 'https://your-app.com')
        self.site_name = os.environ.get('SITE_NAME', 'Word Analysis App')
    
    def _format_messages_for_model(self, model, content, system_prompt):
        """
        Format messages based on the model type with explicit JSON instructions.
        
        Args:
            model (str): The model identifier string
            content (str): The user content to send
            system_prompt (str): The system prompt
            
        Returns:
            list: Properly formatted messages for the specific model
        """
        # Add explicit JSON formatting instructions to all prompts
        json_instructions = """
        IMPORTANT: You must respond with valid JSON only.
        Do not include any explanatory text outside the JSON.
        Do not use code blocks, LaTeX notation, or other formatting.
        Provide raw JSON only.
        """
        
        enhanced_system_prompt = system_prompt + json_instructions
        
        # Format for Google models (Gemini/Gemma)
        if "google" in model.lower():
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{content}\n\n{enhanced_system_prompt}"
                        }
                    ]
                }
            ]
        # Default format for most models
        else:
            return [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": content}
            ]
    
    def _parse_json_response(self, content_json):
        """
        Simplified JSON parsing that tries multiple strategies
        
        Args:
            content_json (str): The response string
            
        Returns:
            dict: Parsed JSON or None
        """
        # First, try direct JSON parsing
        try:
            return json.loads(content_json)
        except json.JSONDecodeError:
            pass
        
        # If that fails, try to extract any JSON-like structure
        try:
            import re
            json_pattern = r'{.*}'
            match = re.search(json_pattern, content_json, re.DOTALL)
            if match:
                json_candidate = match.group(0)
                return json.loads(json_candidate)
        except:
            pass
        
        # If all attempts fail
        print(f"[OpenRouter] Failed to parse response: {content_json[:100]}...")
        return None
    
    def call_model(self, model, messages, max_tokens=500, retry_count=2):
        """
        Call a model with appropriate formatting and error handling
        
        Args:
            model (str): The model to use
            messages (list): Formatted messages
            max_tokens (int): Maximum response tokens
            retry_count (int): Number of retries
            
        Returns:
            dict: Parsed response or None
        """
        for attempt in range(retry_count):
            try:
                print(f"[OpenRouter] Attempt {attempt+1}/{retry_count} - Using model: {model}")
                start_time = time.time()
                
                # Response format is different per model
                response_format = {"type": "json_object"} if "google" not in model.lower() else None
                
                # Set up request parameters
                request_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                    "extra_headers": {
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
                }
                
                # Add response_format parameter only for non-Google models
                if response_format:
                    request_params["response_format"] = response_format
                
                # Add the timeout parameter explicitly for the completion call
                # Set a strict timeout for the completion call
                completion_timeout = httpx.Timeout(30.0, connect=5.0, read=25.0, write=5.0)
                
                # Wrap the API call in a try-except with a timeout parameter
                response = self.client.chat.completions.create(**request_params)
                
                # Record successful usage
                self.usage_tracker.record_usage(model)
                
                api_call_time = time.time() - start_time
                print(f"[OpenRouter] API call completed in {api_call_time:.2f}s using model {model}")
                
                content_json = response.choices[0].message.content
                
                # Check for valid content
                if content_json and len(content_json) > 0:
                    print(f"[OpenRouter] Received response of length {len(content_json)}")
                    result = self._parse_json_response(content_json)
                    
                    if result:
                        # Add model info
                        result["model"] = model
                        return result
                    else:
                        print(f"[OpenRouter] Failed to parse JSON from model {model}")
                else:
                    print(f"[OpenRouter] Empty response, retrying ({attempt+1}/{retry_count})...")
                    time.sleep(1)
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit errors
                if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
                    print(f"[OpenRouter] Rate limit exceeded for model {model}")
                    self.usage_tracker.disable_model(model)
                # Check for timeout errors and log them specifically
                elif "timeout" in error_str or "timed out" in error_str:
                    print(f"[OpenRouter] Request timed out for model {model}: {error_str}")
                    # Disable the model temporarily since it's not responding
                    self.usage_tracker.disable_model(model)
                else:
                    print(f"[OpenRouter] Error calling model: {str(e)}, retrying ({attempt+1}/{retry_count})...")
                
                time.sleep(1)
        
        # All attempts failed
        print(f"[OpenRouter] All {retry_count} attempts failed for model {model}")
        return None