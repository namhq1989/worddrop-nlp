import os
import time
import json
from datetime import datetime, timedelta
import threading

# OpenRouter rate limit constants for free models
# These can be easily modified if the limits change
OPENROUTER_FREE_REQUESTS_PER_MINUTE = 20
OPENROUTER_FREE_REQUESTS_PER_DAY = 200
OPENROUTER_RESET_CHECK_INTERVAL = 10  # Seconds between checking for reset periods

class OpenRouterUsageTracker:
    """
    Class to track usage of OpenRouter models and rotate between them to stay within rate limits.
    Each free model is limited to OPENROUTER_FREE_REQUESTS_PER_MINUTE requests per minute
    and OPENROUTER_FREE_REQUESTS_PER_DAY requests per day.
    """
    
    # Hardcoded list of available OpenRouter models to use
    AVAILABLE_MODELS = [
        # Google models
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-3-4b-it:free",
        
        # Other models
        "deepseek/deepseek-r1-zero:free",
        "qwen/qwq-32b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "microsoft/phi-3-medium-128k-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "mistralai/mistral-nemo:free",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "cognitivecomputations/dolphin3.0-mistral-24b:free",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    ]
    
    def __init__(self):
        """Initialize the usage tracker with hardcoded available models"""
        # Check if rate limits are overridden in environment variables
        self.requests_per_minute = int(os.environ.get('OPENROUTER_REQUESTS_PER_MINUTE', 
                                                   OPENROUTER_FREE_REQUESTS_PER_MINUTE))
        self.requests_per_day = int(os.environ.get('OPENROUTER_REQUESTS_PER_DAY', 
                                                OPENROUTER_FREE_REQUESTS_PER_DAY))
        self.reset_check_interval = int(os.environ.get('OPENROUTER_RESET_CHECK_INTERVAL',
                                                    OPENROUTER_RESET_CHECK_INTERVAL))
        
        # Use hardcoded models
        self.available_models = self.AVAILABLE_MODELS.copy()
        
        if not self.available_models:
            raise ValueError("No OpenRouter models configured.")
        
        print(f"[OpenRouterUsageTracker] Initialized with {len(self.available_models)} models")
        print(f"[OpenRouterUsageTracker] Available models: {', '.join(model.split('/')[0] for model in self.available_models)}")
        print(f"[OpenRouterUsageTracker] Rate limits: {self.requests_per_minute}/minute, {self.requests_per_day}/day")
        
        # Initialize usage tracking
        self.usage = {}
        self.last_reset_day = datetime.now().day
        self.lock = threading.Lock()
        
        # Initialize each model's tracking
        for model in self.available_models:
            self.usage[model] = {
                'day_count': 0,              # Requests today
                'minute_count': 0,           # Requests in the current minute
                'last_minute': datetime.now().minute,
                'last_used': datetime.now() - timedelta(hours=1),  # Start with models being unused for 1 hour
                'enabled': True              # Whether the model is currently enabled
            }
        
        # Start a background thread to reset counters
        self._start_reset_thread()
    
    def _start_reset_thread(self):
        """Start a background thread to periodically reset counters"""
        def reset_thread():
            while True:
                now = datetime.now()
                
                # Reset daily counters at midnight
                if now.day != self.last_reset_day:
                    with self.lock:
                        for model in self.usage:
                            self.usage[model]['day_count'] = 0
                            self.usage[model]['enabled'] = True
                        self.last_reset_day = now.day
                        print(f"[OpenRouterUsageTracker] Reset daily counters at {now}")
                
                # Reset minute counters when the minute changes
                current_minute = now.minute
                for model in self.usage:
                    if self.usage[model]['last_minute'] != current_minute:
                        with self.lock:
                            self.usage[model]['minute_count'] = 0
                            self.usage[model]['last_minute'] = current_minute
                
                # Sleep for the configured interval before checking again
                time.sleep(self.reset_check_interval)
        
        # Start the thread as a daemon so it will exit when the main program exits
        thread = threading.Thread(target=reset_thread, daemon=True)
        thread.start()
    
    def get_next_available_model(self):
        """
        Get the next available model that hasn't exceeded its rate limits.
        Uses a scoring system that prioritizes:
        1. Models that aren't disabled (haven't hit daily limit)
        2. Models with the lowest usage in the current minute
        3. Models that haven't been used recently
        
        Returns:
            str: Model ID to use
        """
        with self.lock:
            now = datetime.now()
            current_minute = now.minute
            
            # Reset minute counters if needed
            for model in self.usage:
                if self.usage[model]['last_minute'] != current_minute:
                    self.usage[model]['minute_count'] = 0
                    self.usage[model]['last_minute'] = current_minute
            
            # Score each model and select the best one
            best_model = None
            best_score = -1
            
            for model in self.available_models:
                # Skip disabled models (reached daily limit)
                if not self.usage[model]['enabled']:
                    continue
                
                # Calculate time since last use in seconds
                time_since_use = (now - self.usage[model]['last_used']).total_seconds()
                
                # Calculate score based on usage and time
                # Higher score is better:
                # - Fewer requests in current minute (max is the per-minute limit)
                # - Longer time since last use
                minute_factor = self.requests_per_minute - self.usage[model]['minute_count']
                time_factor = min(time_since_use / 60, 10)  # Cap at 10 for 10+ minutes unused
                
                score = (minute_factor * 10) + time_factor
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            # If all models are disabled, choose the first one as fallback
            if best_model is None and self.available_models:
                print("[OpenRouterUsageTracker] WARNING: All models have reached daily limits. Using first model anyway.")
                best_model = self.available_models[0]
            
            return best_model
    
    def disable_model(self, model):
        """
        Explicitly disable a model, typically after receiving a rate limit error.
        
        Args:
            model (str): The model ID to disable
        """
        with self.lock:
            if model in self.usage:
                self.usage[model]['enabled'] = False
                print(f"[OpenRouterUsageTracker] Model {model} has been disabled due to rate limiting")
            else:
                # Initialize tracking for new models
                self.usage[model] = {
                    'day_count': self.requests_per_day,  # Set to limit to ensure it stays disabled
                    'minute_count': self.requests_per_minute,
                    'last_minute': datetime.now().minute,
                    'last_used': datetime.now(),
                    'enabled': False
                }
                print(f"[OpenRouterUsageTracker] Added and disabled new model {model}")
    
    def record_usage(self, model):
        """
        Record usage of a model and check if it has reached limits.
        
        Args:
            model (str): The model ID that was used
            
        Returns:
            bool: True if the model is still available, False if it's reached its daily limit
        """
        with self.lock:
            if model not in self.usage:
                # Initialize tracking for new models
                self.usage[model] = {
                    'day_count': 0,
                    'minute_count': 0,
                    'last_minute': datetime.now().minute,
                    'last_used': datetime.now(),
                    'enabled': True
                }
            
            # Update usage counters
            self.usage[model]['day_count'] += 1
            self.usage[model]['minute_count'] += 1
            self.usage[model]['last_used'] = datetime.now()
            
            # Check if model has reached daily limit
            if self.usage[model]['day_count'] >= self.requests_per_day:
                self.usage[model]['enabled'] = False
                print(f"[OpenRouterUsageTracker] Model {model} has reached its daily limit of {self.requests_per_day} requests")
            
            return self.usage[model]['enabled']
    
    def get_usage_stats(self):
        """
        Get current usage statistics for all models.
        
        Returns:
            dict: Usage statistics
        """
        with self.lock:
            stats = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'models': {}
            }
            
            for model in self.usage:
                stats['models'][model] = {
                    'day_count': self.usage[model]['day_count'],
                    'minute_count': self.usage[model]['minute_count'],
                    'last_used': self.usage[model]['last_used'].strftime('%Y-%m-%d %H:%M:%S'),
                    'enabled': self.usage[model]['enabled'],
                    'percent_daily_limit': round((self.usage[model]['day_count'] / self.requests_per_day) * 100, 1),
                    'disabled_reason': 'Reached daily limit' if self.usage[model]['day_count'] >= self.requests_per_day else 
                                      ('Manually disabled' if not self.usage[model]['enabled'] else 'N/A')
                }
            
            return stats
    
    def add_model(self, model):
        """
        Add a new model to the available models list.
        
        Args:
            model (str): The model ID to add
        """
        with self.lock:
            if model not in self.available_models:
                self.available_models.append(model)
                self.usage[model] = {
                    'day_count': 0,
                    'minute_count': 0,
                    'last_minute': datetime.now().minute,
                    'last_used': datetime.now() - timedelta(hours=1),
                    'enabled': True
                }
                print(f"[OpenRouterUsageTracker] Added new model: {model}")
    
    def remove_model(self, model):
        """
        Remove a model from the available models list.
        
        Args:
            model (str): The model ID to remove
        """
        with self.lock:
            if model in self.available_models:
                self.available_models.remove(model)
                if model in self.usage:
                    del self.usage[model]
                print(f"[OpenRouterUsageTracker] Removed model: {model}")