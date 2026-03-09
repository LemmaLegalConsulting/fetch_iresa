#!/usr/bin/env python3
"""
Test script to verify JSON parsing robustness improvements.
Run this to demonstrate the enhanced recovery capabilities.
"""

# Prevent pytest from collecting this script as a test module.
__test__ = False

import json
import sys
sys.path.insert(0, '/home/quinten/fetch')

from app.utils.json_helpers import extract_json_from_fenced_code, parse_json_from_llm_response

def test_case(name, content, expected_key=None):
    """Test a single JSON parsing case."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"Input: {content[:80]}{'...' if len(content) > 80 else ''}")
    print(f"{'='*60}")
    
    try:
        result = parse_json_from_llm_response(content)
        print(f"✅ SUCCESS")
        if expected_key:
            if expected_key in result:
                print(f"   Found expected key '{expected_key}': {result[expected_key]}")
            else:
                print(f"   ⚠️  Expected key '{expected_key}' not found. Keys: {list(result.keys())}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Run all test cases."""
    print("JSON Parsing Robustness Test Suite")
    print("=" * 60)
    
    test_cases = [
        # Test 1: Standard format (should always work)
        ("Standard format with newlines", 
         '```json\n{"labels": ["test"]}\n```',
         "labels"),
        
        # Test 2: No newlines (Pattern 2)
        ("Format without newlines",
         '```json{"labels": ["test"]}```',
         "labels"),
        
        # Test 3: Trailing comma (common LLM error)
        ("Trailing comma in array",
         '{"labels": ["test",], "questions": []}',
         "labels"),
        
        # Test 4: Trailing comma in object
        ("Trailing comma in object",
         '{"labels": ["test"], "questions": [],}',
         "labels"),
        
        # Test 5: Backtick variation
        ("Backtick fence variation",
         '````{"labels": ["test"]}````',
         "labels"),
        
        # Test 6: JSON in text
        ("JSON mixed with text",
         'Here is the result: {"labels": ["test"], "questions": []} and some more text',
         "labels"),
        
        # Test 7: No fences (plain JSON)
        ("Plain JSON without fences",
         '{"labels": ["test"], "questions": []}',
         "labels"),
        
        # Test 8: Multiple issues
        ("Multiple issues: fences + trailing comma",
         '```json\n{"labels": ["test",], "questions": [],}\n```',
         "labels"),
    ]
    
    passed = 0
    failed = 0
    
    for name, content, expected_key in test_cases:
        if test_case(name, content, expected_key):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    if failed == 0:
        print("✅ All tests passed! JSON parsing robustness improvements working.")
    else:
        print(f"⚠️  {failed} test(s) failed.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
