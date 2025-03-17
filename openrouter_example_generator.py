from openrouter_handler import OpenRouterHandler

class OpenRouterExampleGenerator(OpenRouterHandler):
    """Example generator using OpenRouter API"""
    
    def __init__(self, usage_tracker):
        """Initialize the Example Generator"""
        super().__init__(usage_tracker)
        
        # Define the system prompt
        self.system_prompt = """
        You are a creative language teacher and expert linguist. For each word, generate exactly one diverse and original example sentence for each level (Beginner, Intermediate, Advanced).
        
        STRICT REQUIREMENTS:
        1. Create truly varied and creative examples - avoid formulaic patterns
        2. Beginner examples should be simple but not always starting with "I" or "We"
        3. Intermediate examples should use diverse grammatical structures
        4. Advanced examples should NOT always start with "Having..." - use varied complex structures
        5. VERY IMPORTANT: Each example MUST include the EXACT word given, not derivatives, roots, or related words
        6. For each example, identify the exact form of the word used, which MUST MATCH the input word
        7. Avoid repeating the same sentence structures across different words
        8. Do not make up words or use non-standard English
        9. If the word is a noun, use it as a noun; if it's a verb, use it as a verb, etc.
        
        Output in JSON format with exactly two fields:
        - "word": The exact input word
        - "examples": An object with three keys: "beginner", "intermediate", and "advanced", each containing an object with "example" and "word" fields
        """
    
    def generate_examples(self, word, retry_count=3):
        """Generate examples for a given word with retry logic using OpenRouter"""
        # Store original word for validation
        original_word = word.lower()
        
        # Create the content prompt
        content = f"Generate creative example sentences for the word: {word}"
        
        # Get the best model to use for this request
        model = self.usage_tracker.get_next_available_model()
        if not model:
            raise ValueError("No OpenRouter models available. All models may have reached their daily limits.")
        
        # Format messages based on the model type
        messages = self._format_messages_for_model(model, content, self.system_prompt)
        
        # Call the model
        result = self.call_model(model, messages, max_tokens=250, retry_count=retry_count)
        
        if result:
            # Validate the structure and content
            if "examples" in result and "word" in result:
                if all(level in result["examples"] for level in ["beginner", "intermediate", "advanced"]):
                    # Ensure the top-level word matches the original
                    result["word"] = original_word
                    
                    # Add provider to the response
                    result["provider"] = "openrouter"
                    
                    # Basic validation of example structure
                    for level in ["beginner", "intermediate", "advanced"]:
                        example_data = result["examples"][level]
                        if not isinstance(example_data, dict) or "example" not in example_data or "word" not in example_data:
                            print(f"[OpenRouterExampleGenerator] Invalid example structure for {level}")
                            result["examples"][level] = {
                                "example": f"The {level} student studied the word '{original_word}' carefully.",
                                "word": original_word
                            }
                    
                    print(f"[OpenRouterExampleGenerator] Successfully generated examples for '{word}'")
                    return result
        
        # Return a default response if all retries fail
        print(f"[OpenRouterExampleGenerator] Failed to generate valid examples, returning default response for '{word}'")
        return {
            "word": original_word,
            "examples": {
                "beginner": {
                    "example": f"The teacher asked us to use {original_word} in a simple sentence.",
                    "word": original_word
                },
                "intermediate": {
                    "example": f"Students were required to understand how {original_word} is used in different contexts.",
                    "word": original_word
                },
                "advanced": {
                    "example": f"Researchers examining linguistic patterns found that {original_word} appears frequently in academic literature.",
                    "word": original_word
                }
            },
            "model": "fallback",
            "provider": "openrouter"
        }