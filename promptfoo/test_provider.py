#!/usr/bin/env python3
"""Simple test provider to verify Promptfoo can call Python providers."""

def call_api(prompt, options, context):
    """Simple test implementation."""
    return {
        "output": '{"test": "success", "prompt": "' + prompt[:50] + '"}'
    }
