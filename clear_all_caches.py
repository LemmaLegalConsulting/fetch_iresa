#!/usr/bin/env python3
"""
Clear all caches used by the classification system.

WHEN TO USE THIS:
- Once before testing a NEW HYPOTHESIS (before first eval of that hypothesis)
- NOT needed between individual test runs (the LRU cache is auto-cleared each invocation)
- The other 3 cache layers (rendered prompts, service responses, promptfoo results)
  persist across evaluations and are cleared here

CACHES CLEARED:
1. Python's LRU cache for prompt templates (auto-cleared in provider, but cleared here too for safety)
2. In-memory rendered prompt cache dictionary
3. Service response cache (diskcache: ./cache/provider_responses_cache/)
4. Promptfoo evaluation cache (.promptfoo/cache/cache.json)
"""

import os
import shutil
from pathlib import Path

def clear_all_caches():
    """Clear all cache layers."""
    
    # Change to repo root
    if os.path.isfile(__file__):
        repo_root = os.path.dirname(os.path.abspath(__file__))
    else:
        repo_root = os.path.abspath(".")
    
    os.chdir(repo_root)
    if repo_root not in __import__("sys").path:
        __import__("sys").path.insert(0, repo_root)
    
    print("🔄 Clearing all caches before new hypothesis test...\n")
    
    # Layer 1 & 2: Python LRU cache AND rendered prompt cache
    print("  1-2. Clearing Python prompt caches (LRU + rendered dict)...")
    from app.providers.base import clear_all_prompt_caches
    clear_all_prompt_caches()
    print("     ✓ All Python prompt caches cleared")
    
    # Layer 3: Service response cache (diskcache)
    print("  3. Clearing service response cache (diskcache)...")
    cache_dir = Path("./cache/provider_responses_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"     ✓ Deleted {cache_dir}")
    else:
        print(f"     - {cache_dir} does not exist (already clean)")
    
    # Layer 4: Promptfoo evaluation cache
    print("  4. Clearing promptfoo evaluation cache...")
    promptfoo_cache = Path("./.promptfoo/cache/cache.json")
    if promptfoo_cache.exists():
        promptfoo_cache.unlink()
        print(f"     ✓ Deleted {promptfoo_cache}")
    else:
        print(f"     - {promptfoo_cache} does not exist (already clean)")
    
    print("\n✅ All caches cleared!\n")
    print("📋 Ready to run: promptfoo eval -c promptfoo/followup_questions_eval.yaml --no-cache")
    print("   (The --no-cache flag is REQUIRED to bypass promptfoo's built-in caching)\n")

if __name__ == "__main__":
    clear_all_caches()
