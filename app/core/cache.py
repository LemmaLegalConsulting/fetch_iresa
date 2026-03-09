from diskcache import Cache
import os

_cache = None


def get_cache():
    global _cache
    if _cache is None:
        cache_dir = "./cache"
        _cache = Cache(os.path.join(cache_dir, "provider_responses_cache"))
    return _cache
