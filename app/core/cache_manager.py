import logging
from typing import Any, Optional, Dict, Tuple
import asyncio
from datetime import datetime, timedelta
import json
from functools import wraps

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Centralized caching system for the application.
    Handles all caching operations with proper TTL and type safety.
    """
    _instance = None
    _cache: Dict[str, Tuple[datetime, Any]] = {}
    _cache_ttl = timedelta(minutes=5)
    _cache_lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache if it exists and hasn't expired"""
        async with self._cache_lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if datetime.utcnow() - timestamp < self._cache_ttl:
                    return value
                del self._cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None):
        """Set a value in cache with optional custom TTL"""
        async with self._cache_lock:
            self._cache[key] = (datetime.utcnow(), value)

    async def delete(self, key: str):
        """Delete a value from cache"""
        async with self._cache_lock:
            if key in self._cache:
                del self._cache[key]

    async def clear(self):
        """Clear all cached values"""
        async with self._cache_lock:
            self._cache.clear()

    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key from prefix and arguments"""
        key_parts = [prefix]
        
        # Add positional arguments
        if args:
            key_parts.extend(str(arg) for arg in args)
        
        # Add keyword arguments in sorted order
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}:{v}" for k, v in sorted_kwargs)
        
        return ":".join(key_parts)

    def cache_decorator(self, prefix: str, ttl: Optional[timedelta] = None):
        """
        Decorator for caching function results.
        Usage:
            @cache_manager.cache_decorator("user", timedelta(minutes=5))
            async def get_user(user_id: str):
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.generate_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # If not in cache, call function
                result = await func(*args, **kwargs)
                
                # Cache the result
                await self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator

# Create singleton instance
_cache_manager = None

async def get_cache_manager() -> CacheManager:
    """Get or initialize singleton instance of CacheManager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager 