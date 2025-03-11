import os
import json
import time
from openai import OpenAI

class ExampleGenerator:
    def __init__(self):
        """Initialize the Example Generator with DeepSeek API"""
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        
        # Define the static prefix with system prompt and few-shot examples for caching
        self.system_prompt = """
        You are a creative language teacher. For each word, generate exactly one diverse and original example sentence for each level (Beginner, Intermediate, Advanced).
        
        Requirements:
        1. Create truly varied and creative examples - avoid formulaic patterns
        2. Beginner examples should be simple but not always starting with "I" or "We"
        3. Intermediate examples should use diverse grammatical structures
        4. Advanced examples should NOT always start with "Having..." - use varied complex structures
        5. The word in each example doesn't need to be in the base form - it can be any variation
        6. For each example, identify the exact form of the word used
        7. Avoid repeating the same sentence structures across different words
        8. Output in JSON format only
        
        EXAMPLE INPUT:
        Word: eat

        EXAMPLE JSON OUTPUT:
        {
            "word": "eat",
            "examples": {
                "beginner": {
                    "example": "Most animals eat plants or other animals.",
                    "word": "eat"
                },
                "intermediate": {
                    "example": "By the time we arrived, they had already eaten dinner.",
                    "word": "eaten"
                },
                "advanced": {
                    "example": "The critic's scathing review ate away at his confidence in the new novel.",
                    "word": "ate"
                }
            }
        }
        """
        
        # Setup static messages for caching benefit
        self.static_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Word: play"},
            {"role": "assistant", "content": json.dumps({
                "word": "play",
                "examples": {
                    "beginner": {
                        "example": "Dogs like to play with balls.",
                        "word": "play"
                    },
                    "intermediate": {
                        "example": "The director wanted someone who could play both comedic and dramatic roles.",
                        "word": "play"
                    },
                    "advanced": {
                        "example": "Market forces often play a significant role in determining economic outcomes.",
                        "word": "play"
                    }
                }
            })},
            {"role": "user", "content": "Word: smile"},
            {"role": "assistant", "content": json.dumps({
                "word": "smile",
                "examples": {
                    "beginner": {
                        "example": "Her smile brightened the room.",
                        "word": "smile"
                    },
                    "intermediate": {
                        "example": "Despite the pain, he managed to smile at the nurse.",
                        "word": "smile"
                    },
                    "advanced": {
                        "example": "Fortune smiled upon the small village when a rich deposit of minerals was discovered nearby.",
                        "word": "smiled"
                    }
                }
            })}
        ]
    
    def generate_examples(self, word, retry_count=3):
        """Generate examples for a given word with retry logic"""
        # Add a randomization element to encourage variety
        variation_seeds = [
            f"Create diverse examples for the word '{word}'. Avoid starting advanced examples with 'Having'.",
            f"Generate unique example sentences using '{word}' at different levels.",
            f"Show creative ways to use '{word}' in sentences of varying complexity."
        ]
        
        # Pick a random seed for this request
        import random
        seed = random.choice(variation_seeds)
        
        messages = self.static_messages.copy()
        messages.append({"role": "user", "content": f"{seed}\nWord: {word}"})
        
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    response_format={'type': 'json_object'},
                    max_tokens=500,
                    temperature=1.5
                )
                
                content = response.choices[0].message.content
                
                # Check if we got valid content
                if content and len(content) > 0:
                    result = json.loads(content)
                    
                    # Log cache hit information if available
                    if hasattr(response.usage, "prompt_cache_hit_tokens"):
                        cache_hit = response.usage.prompt_cache_hit_tokens
                        cache_miss = response.usage.prompt_cache_miss_tokens
                        print(f"Word: {word}, Cache hit: {cache_hit}, Cache miss: {cache_miss}")
                    
                    return result
                else:
                    print(f"Empty response for word '{word}', retrying ({attempt+1}/{retry_count})...")
                    time.sleep(1)
            except Exception as e:
                print(f"Error processing word '{word}': {str(e)}, retrying ({attempt+1}/{retry_count})...")
                time.sleep(1)
        
        # Return a default response if all retries fail
        return {
            "word": word,
            "examples": {
                "beginner": {
                    "example": f"Failed to generate example for {word}.",
                    "word": word
                },
                "intermediate": {
                    "example": f"Failed to generate example for {word}.",
                    "word": word
                },
                "advanced": {
                    "example": f"Failed to generate example for {word}.",
                    "word": word
                }
            }
        }