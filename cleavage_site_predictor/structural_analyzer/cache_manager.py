import os
import pickle
import hashlib
from typing import Dict, Any, Optional, Union
from abc import ABC


class CacheManager(ABC):
    """
    CacheManager is a parent class for all cache managers.
    It supports two levels of caching: RAM cache and disk cache.
    """
    
    def __init__(self, cache_dir: str = "structure_temp/cache", 
                 use_disk_cache: bool = True, use_ram_cache: bool = True):
        """
        Initialize the CacheManager
        
        Args:
            cache_dir: disk cache directory
            use_disk_cache: whether to enable disk cache
            use_ram_cache: whether to enable RAM cache
        """
        # cache configuration
        self.cache_dir = cache_dir
        self.use_disk_cache = use_disk_cache
        self.use_ram_cache = use_ram_cache
        
        # create disk cache directory
        if self.use_disk_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        # initialize RAM cache dictionary
        if self.use_ram_cache:
            self._ram_cache = {}  # {cache_key: cached_data}
        else:
            self._ram_cache = None
    
    def _generate_cache_key(self, **kwargs) -> str:
        """
        Generate a cache key
        
        Args:
            **kwargs: key-value pairs
            
        Returns:
            str: MD5 hash cache key
        """
        # filter None values and sort to ensure consistency
        key_parts = []
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}={value}")
        
        key_string = "&".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_to_disk(self, cache_key: str, data: Any, cache_type: str) -> bool:
        """
        Save data to disk cache
        
        Args:
            cache_key: cache key
            data: data to cache
            cache_type: cache type (used for file name prefix)
            
        Returns:
            bool: whether the data is saved successfully
        """
        if not self.use_disk_cache:
            return False
            
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_type}_{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")
            return False
    
    def _load_from_disk(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """
        Load data from disk cache
        
        Args:
            cache_key: cache key
            cache_type: cache type
            
        Returns:
            Any: cached data, if not exists, return None
        """
        if not self.use_disk_cache:
            return None
            
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_type}_{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache from disk: {e}")
        return None
    
    def _get_cached_data(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """
        Get data from two levels of cache
        
        Args:
            cache_key: cache key
            cache_type: cache type
            
        Returns:
            Any: cached data, if not exists, return None
        """
        # 1. check RAM cache (fastest)
        if self.use_ram_cache and self._ram_cache and cache_key in self._ram_cache:
            print(f"📦 use RAM cache: {cache_type}_{cache_key[:8]}...")
            return self._ram_cache[cache_key]
        
        # 2. check disk cache (medium speed)
        disk_data = self._load_from_disk(cache_key, cache_type)
        if disk_data is not None:
            print(f"💾 use disk cache: {cache_type}_{cache_key[:8]}...")
            
            # promote disk cache result to RAM cache
            if self.use_ram_cache and self._ram_cache is not None:
                self._ram_cache[cache_key] = disk_data
                
            return disk_data
        
        return None
    
    def _cache_data(self, cache_key: str, data: Any, cache_type: str) -> None:
        """
        Save data to two levels of cache
        
        Args:
            cache_key: cache key
            data: data to cache
            cache_type: cache type
        """
        # save to RAM cache
        if self.use_ram_cache and self._ram_cache is not None:
            self._ram_cache[cache_key] = data
        
        # save to disk cache
        self._save_to_disk(cache_key, data, cache_type)
    
    def clear_cache(self, cache_type: str = "all") -> None:
        """
        Clear cache
        
        Args:
            cache_type: cache type ("ram", "disk", "all")
        """
        if cache_type in ["all", "ram"] and self.use_ram_cache:
            if self._ram_cache is not None:
                self._ram_cache.clear()
            print(" RAM cache cleared")
            
        if cache_type in ["all", "disk"] and self.use_disk_cache:
            try:
                import shutil
                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                    os.makedirs(self.cache_dir, exist_ok=True)
                print(" disk cache cleared")
            except Exception as e:
                print(f"Warning: Failed to clear disk cache: {e}")
 