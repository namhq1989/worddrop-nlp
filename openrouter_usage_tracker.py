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
OPENROUTER_TIMEOUT_RETRY_DELAY = 60  # Wait 1 minute before retrying a model that timed out

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
        "mistralai/mistral-nemo:free",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
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
        self.timeout_retry_delay = int(os.environ.get('OPENROUTER_TIMEOUT_RETRY_DELAY',
                                                   OPENROUTER_TIMEOUT_RETRY_DELAY))
        
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
                'enabled': True,              # Whether the model is currently enabled
                'disabled_reason': None,      # Why the model is disabled
                'disabled_until': None        # When to re-enable the model
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
                            
                            # Only re-enable if it was disabled due to rate limiting
                            if self.usage[model]['disabled_reason'] == 'daily_limit':
                                self.usage[model]['enabled'] = True
                                self.usage[model]['disabled_reason'] = None
                                self.usage[model]['disabled_until'] = None
                                
                        self.last_reset_day = now.day
                        print(f"[OpenRouterUsageTracker] Reset daily counters at {now}")
                
                # Reset minute counters when the minute changes
                current_minute = now.minute
                for model in self.usage:
                    if self.usage[model]['last_minute'] != current_minute:
                        with self.lock:
                            self.usage[model]['minute_count'] = 0
                            self.usage[model]['last_minute'] = current_minute
                
                # Check if any timed-out models can be re-enabled
                for model in self.usage:
                    if (not self.usage[model]['enabled'] and 
                        self.usage[model]['disabled_reason'] == 'timeout' and
                        self.usage[model]['disabled_until'] and 
                        now >= self.usage[model]['disabled_until']):
                        with self.lock:
                            self.usage[model]['enabled'] = True
                            self.usage[model]['disabled_reason'] = None
                            self.usage[model]['disabled_until'] = None
                            print(f"[OpenRouterUsageTracker] Re-enabling model {model} after timeout period")
                
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
            
            # First pass - prioritize Mistral models before Google models to reduce timeouts
            for model in self.available_models:
                # Skip disabled models
                if not self.usage[model]['enabled']:
                    continue
                
                # Calculate time since last use in seconds
                time_since_use = (now - self.usage[model]['last_used']).total_seconds()
                
                # Calculate score based on usage and time
                # Higher score is better:
                # - Fewer requests in current minute (max is the per-minute limit)
                # - Longer time since last use
                # - Add a bonus for Mistral models since they seem more reliable
                minute_factor = self.requests_per_minute - self.usage[model]['minute_count']
                time_factor = min(time_since_use / 60, 10)  # Cap at 10 for 10+ minutes unused
                
                # Add a bonus for Mistral models to prioritize them over Google models
                provider_bonus = 100 if 'mistral' in model.lower() else 0
                
                score = (minute_factor * 10) + time_factor + provider_bonus
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            # If all models are disabled, choose a Mistral model as fallback
            if best_model is None and self.available_models:
                # Try to find a Mistral model first
                for model in self.available_models:
                    if 'mistral' in model.lower():
                        print(f"[OpenRouterUsageTracker] WARNING: All models have restrictions. Using {model} as fallback.")
                        return model
                
                # If no Mistral model found, use the first available
                best_model = self.available_models[0]
                print(f"[OpenRouterUsageTracker] WARNING: All models have restrictions. Using {best_model} as fallback.")
            
            return best_model
    
    def disable_model(self, model, reason='unknown'):
        """
        Explicitly disable a model, typically after receiving a rate limit error.
        
        Args:
            model (str): The model ID to disable
            reason (str): Why the model is being disabled
        """
        with self.lock:
            if model in self.usage:
                self.usage[model]['enabled'] = False
                self.usage[model]['disabled_reason'] = reason
                
                # Set a re-enable time if it's a timeout
                if reason == 'timeout':
                    self.usage[model]['disabled_until'] = datetime.now() + timedelta(seconds=self.timeout_retry_delay)
                    print(f"[OpenRouterUsageTracker] Model {model} has been disabled due to timeout. Will retry in {self.timeout_retry_delay/60} minutes.")
                else:
                    self.usage[model]['disabled_reason'] = 'daily_limit'
                    print(f"[OpenRouterUsageTracker] Model {model} has been disabled due to rate limiting")
            else:
                # Initialize tracking for new models
                disabled_until = None
                if reason == 'timeout':
                    disabled_until = datetime.now() + timedelta(seconds=self.timeout_retry_delay)
                
                self.usage[model] = {
                    'day_count': self.requests_per_day if reason == 'daily_limit' else 0,
                    'minute_count': self.requests_per_minute if reason == 'rate_limit' else 0,
                    'last_minute': datetime.now().minute,
                    'last_used': datetime.now(),
                    'enabled': False,
                    'disabled_reason': reason,
                    'disabled_until': disabled_until
                }
                print(f"[OpenRouterUsageTracker] Added and disabled new model {model} due to {reason}")
    
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
                    'enabled': True,
                    'disabled_reason': None,
                    'disabled_until': None
                }
            
            # Update usage counters
            self.usage[model]['day_count'] += 1
            self.usage[model]['minute_count'] += 1
            self.usage[model]['last_used'] = datetime.now()
            
            # Check if model has reached daily limit
            if self.usage[model]['day_count'] >= self.requests_per_day:
                self.usage[model]['enabled'] = False
                self.usage[model]['disabled_reason'] = 'daily_limit'
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
                    'disabled_reason': self.usage[model]['disabled_reason'] or 'N/A',
                    'disabled_until': self.usage[model]['disabled_until'].strftime('%Y-%m-%d %H:%M:%S') if self.usage[model]['disabled_until'] else 'N/A'
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
                    'enabled': True,
                    'disabled_reason': None,
                    'disabled_until': None
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
                
    def disable_provider(self, provider_name):
        """
        Disable all models from a specific provider (e.g., 'google', 'mistralai')
        
        Args:
            provider_name (str): The provider name to disable
        """
        with self.lock:
            disabled_count = 0
            for model in self.usage:
                if provider_name.lower() in model.lower():
                    if self.usage[model]['enabled']:
                        self.usage[model]['enabled'] = False
                        self.usage[model]['disabled_reason'] = 'provider_issue'
                        self.usage[model]['disabled_until'] = datetime.now() + timedelta(seconds=self.timeout_retry_delay)
                        disabled_count += 1
            
            if disabled_count > 0:
                print(f"[OpenRouterUsageTracker] Disabled {disabled_count} models from provider {provider_name}")
                
    def enable_provider(self, provider_name):
        """
        Enable all models from a specific provider (e.g., 'google', 'mistralai')
        
        Args:
            provider_name (str): The provider name to enable
        """
        with self.lock:
            enabled_count = 0
            for model in self.usage:
                if provider_name.lower() in model.lower():
                    if not self.usage[model]['enabled'] and self.usage[model]['disabled_reason'] == 'provider_issue':
                        self.usage[model]['enabled'] = True
                        self.usage[model]['disabled_reason'] = None
                        self.usage[model]['disabled_until'] = None
                        enabled_count += 1
            
            if enabled_count > 0:
                print(f"[OpenRouterUsageTracker] Enabled {enabled_count} models from provider {provider_name}")