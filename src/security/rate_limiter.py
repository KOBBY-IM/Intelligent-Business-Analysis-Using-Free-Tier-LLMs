"""
Rate Limiting Utilities

This module provides rate limiting functionality to prevent API abuse
and ensure fair usage across all users.
"""

import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls and user actions"""
    
    def __init__(self):
        """Initialize the rate limiter"""
        self.request_history: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, float] = {}
        
        # Default limits
        self.default_limits = {
            'api_calls_per_minute': 60,
            'api_calls_per_hour': 1000,
            'user_queries_per_hour': 100,
            'file_uploads_per_hour': 10,
            'evaluation_submissions_per_hour': 50
        }
        
        # Block duration in seconds
        self.block_duration = 3600  # 1 hour
    
    def is_allowed(self, 
                   identifier: str, 
                   action_type: str = 'api_calls_per_minute',
                   custom_limit: Optional[int] = None) -> bool:
        """
        Check if an action is allowed based on rate limits
        
        Args:
            identifier: User identifier (IP, session ID, etc.)
            action_type: Type of action being rate limited
            custom_limit: Custom limit override
            
        Returns:
            True if action is allowed, False otherwise
        """
        # Check if IP is blocked
        if identifier in self.blocked_ips:
            if time.time() - self.blocked_ips[identifier] < self.block_duration:
                logger.warning(f"Rate limit exceeded for {identifier}, still blocked")
                return False
            else:
                # Unblock after duration
                del self.blocked_ips[identifier]
        
        # Get limit for action type
        limit = custom_limit or self.default_limits.get(action_type, 60)
        
        # Get current timestamp
        now = time.time()
        
        # Get request history for this identifier and action
        key = f"{identifier}:{action_type}"
        history = self.request_history[key]
        
        # Remove old entries (older than 1 hour for hourly limits, 1 minute for minute limits)
        if 'per_hour' in action_type:
            cutoff = now - 3600
        else:
            cutoff = now - 60
        
        # Clean old entries
        while history and history[0] < cutoff:
            history.popleft()
        
        # Check if limit exceeded
        if len(history) >= limit:
            logger.warning(f"Rate limit exceeded for {identifier}: {action_type}")
            self.blocked_ips[identifier] = now
            return False
        
        # Add current request
        history.append(now)
        
        return True
    
    def get_remaining_requests(self, 
                              identifier: str, 
                              action_type: str = 'api_calls_per_minute',
                              custom_limit: Optional[int] = None) -> int:
        """
        Get remaining requests for an identifier
        
        Args:
            identifier: User identifier
            action_type: Type of action
            custom_limit: Custom limit override
            
        Returns:
            Number of remaining requests
        """
        limit = custom_limit or self.default_limits.get(action_type, 60)
        key = f"{identifier}:{action_type}"
        history = self.request_history[key]
        
        # Clean old entries
        now = time.time()
        if 'per_hour' in action_type:
            cutoff = now - 3600
        else:
            cutoff = now - 60
        
        while history and history[0] < cutoff:
            history.popleft()
        
        return max(0, limit - len(history))
    
    def get_reset_time(self, 
                      identifier: str, 
                      action_type: str = 'api_calls_per_minute') -> Optional[datetime]:
        """
        Get the time when rate limit resets
        
        Args:
            identifier: User identifier
            action_type: Type of action
            
        Returns:
            Reset time or None if no requests made
        """
        key = f"{identifier}:{action_type}"
        history = self.request_history[key]
        
        if not history:
            return None
        
        # Get the oldest request
        oldest_request = min(history)
        
        # Calculate reset time
        if 'per_hour' in action_type:
            reset_time = oldest_request + 3600
        else:
            reset_time = oldest_request + 60
        
        return datetime.fromtimestamp(reset_time)
    
    def reset_limits(self, identifier: str, action_type: Optional[str] = None):
        """
        Reset rate limits for an identifier
        
        Args:
            identifier: User identifier
            action_type: Specific action type to reset (None for all)
        """
        if action_type:
            key = f"{identifier}:{action_type}"
            if key in self.request_history:
                del self.request_history[key]
        else:
            # Reset all actions for this identifier
            keys_to_remove = [key for key in self.request_history.keys() 
                             if key.startswith(f"{identifier}:")]
            for key in keys_to_remove:
                del self.request_history[key]
        
        # Remove from blocked list
        if identifier in self.blocked_ips:
            del self.blocked_ips[identifier]
        
        logger.info(f"Reset rate limits for {identifier}")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get rate limiter statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_identifiers': len(set(key.split(':')[0] for key in self.request_history.keys())),
            'total_actions': len(self.request_history),
            'blocked_identifiers': len(self.blocked_ips),
            'limits': self.default_limits.copy()
        }
        
        return stats


class APIRateLimiter:
    """Specialized rate limiter for API calls"""
    
    def __init__(self, provider_name: str):
        """
        Initialize API rate limiter
        
        Args:
            provider_name: Name of the API provider
        """
        self.provider_name = provider_name
        self.rate_limiter = RateLimiter()
        
        # Provider-specific limits
        self.provider_limits = {
            'groq': {
                'requests_per_minute': 100,
                'requests_per_hour': 1000
            },
            'gemini': {
                'requests_per_minute': 60,
                'requests_per_hour': 1500
            },
            'huggingface': {
                'requests_per_minute': 30,
                'requests_per_hour': 500
            }
        }
    
    def check_api_limit(self, identifier: str) -> bool:
        """
        Check if API call is allowed
        
        Args:
            identifier: User identifier
            
        Returns:
            True if allowed, False otherwise
        """
        limits = self.provider_limits.get(self.provider_name, {})
        
        # Check per-minute limit
        per_minute_allowed = self.rate_limiter.is_allowed(
            identifier, 
            'requests_per_minute',
            limits.get('requests_per_minute', 60)
        )
        
        if not per_minute_allowed:
            return False
        
        # Check per-hour limit
        per_hour_allowed = self.rate_limiter.is_allowed(
            identifier,
            'requests_per_hour', 
            limits.get('requests_per_hour', 1000)
        )
        
        return per_hour_allowed
    
    def get_remaining_api_calls(self, identifier: str) -> Dict[str, int]:
        """
        Get remaining API calls for an identifier
        
        Args:
            identifier: User identifier
            
        Returns:
            Dictionary with remaining calls per time period
        """
        limits = self.provider_limits.get(self.provider_name, {})
        
        return {
            'per_minute': self.rate_limiter.get_remaining_requests(
                identifier, 'requests_per_minute', 
                limits.get('requests_per_minute', 60)
            ),
            'per_hour': self.rate_limiter.get_remaining_requests(
                identifier, 'requests_per_hour',
                limits.get('requests_per_hour', 1000)
            )
        } 