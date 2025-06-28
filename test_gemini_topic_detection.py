#!/usr/bin/env python3
"""
Test Gemini-based Topic Detection vs Keyword-based
Demonstrates how to use Gemini for more intelligent topic classification
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.modules.context.context_manager import ContextManager
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.core.config import settings

async def test_gemini_vs_keyword_topic_detection():
    """Test Gemini-based topic detection vs keyword-based approach"""
    print("🤖 Gemini vs Keyword Topic Detection Test")
    print("=" * 50)
    
    try:
        # Initialize both systems
        manager = ContextManager()
        
        # Check if Gemini is available
        if not (hasattr(settings, 'GOOGLE_API_KEY') and settings.GOOGLE_API_KEY):
            print("⚠️ Gemini API key not found. Using keyword-only comparison.")
            use_gemini = False
        else:
            try:
                gemini = await get_gemini_wrapper()
                await gemini.initialize()
                use_gemini = True
                print("✓ Gemini wrapper initialized successfully")
            except Exception as e:
                print(f"⚠️ Gemini initialization failed: {e}")
                use_gemini = False
        
        # Test utterances with varying complexity
        test_utterances = [
            {
                "text": "I'm feeling really drowsy, can you find a place to rest?",
                "expected": "safety",
                "complexity": "simple"
            },
            {
                "text": "What's the fastest route to avoid this traffic jam on I-95?", 
                "expected": "navigation",
                "complexity": "simple"
            },
            {
                "text": "Can you remind me to call my doctor about the test results when we get home?",
                "expected": "memory", 
                "complexity": "medium"
            },
            {
                "text": "The kids are getting restless in the back seat, maybe some music would help?",
                "expected": "entertainment",
                "complexity": "medium"
            },
            {
                "text": "I'm starving and the baby needs a diaper change, where's a good family restaurant with changing facilities?",
                "expected": "food",
                "complexity": "complex"
            },
            {
                "text": "Should we take the scenic route through the mountains or the faster highway considering the weather forecast shows rain?",
                "expected": "navigation",  # Could also be weather
                "complexity": "complex"
            }
        ]
        
        print("\nComparing topic detection methods:")
        print("-" * 70)
        
        keyword_correct = 0
        gemini_correct = 0
        total_tests = len(test_utterances)
        
        for i, test_case in enumerate(test_utterances, 1):
            utterance = test_case["text"]
            expected = test_case["expected"]
            complexity = test_case["complexity"]
            
            print(f"\nTest {i} ({complexity} complexity):")
            print(f"Utterance: '{utterance}'")
            print(f"Expected: {expected}")
            
            # Keyword-based detection
            keyword_topic, keyword_confidence = manager.detect_topic_with_confidence(utterance)
            keyword_match = keyword_topic == expected
            if keyword_match:
                keyword_correct += 1
            
            print(f"  Keyword: {keyword_topic} (conf: {keyword_confidence:.2f}) {'✓' if keyword_match else '✗'}")
            
            # Gemini-based detection
            if use_gemini:
                try:
                    gemini_topic, gemini_confidence = await detect_topic_with_gemini(gemini, utterance)
                    gemini_match = gemini_topic == expected
                    if gemini_match:
                        gemini_correct += 1
                    
                    print(f"  Gemini:  {gemini_topic} (conf: {gemini_confidence:.2f}) {'✓' if gemini_match else '✗'}")
                    
                except Exception as e:
                    print(f"  Gemini:  ERROR - {e}")
            else:
                print(f"  Gemini:  SKIPPED (not available)")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 RESULTS SUMMARY")
        print(f"Keyword-based accuracy: {keyword_correct}/{total_tests} ({keyword_correct/total_tests*100:.1f}%)")
        
        if use_gemini:
            print(f"Gemini-based accuracy: {gemini_correct}/{total_tests} ({gemini_correct/total_tests*100:.1f}%)")
            
            if gemini_correct > keyword_correct:
                print("🏆 Gemini outperformed keyword-based detection")
            elif keyword_correct > gemini_correct:
                print("🏆 Keyword-based detection outperformed Gemini")
            else:
                print("🤝 Both methods performed equally")
        else:
            print("Gemini comparison: SKIPPED")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def detect_topic_with_gemini(gemini, utterance: str) -> tuple[str, float]:
    """Use Gemini to detect topic and confidence"""
    
    prompt = f"""
    You are a topic classifier for an in-car AI assistant. Classify the following utterance into ONE of these topics:

    TOPICS:
    - navigation: directions, routes, traffic, driving, locations
    - safety: tiredness, drowsiness, rest, alerts, emergencies
    - memory: reminders, appointments, scheduling, remembering
    - food: restaurants, hunger, eating, dining
    - entertainment: music, games, movies, fun activities
    - weather: forecast, conditions, temperature, rain, snow
    - general: other conversations, greetings, casual talk

    UTTERANCE: "{utterance}"

    Respond with ONLY the topic name and confidence score (0.0-1.0) separated by a comma.
    Example: navigation,0.85

    Response:"""
    
    try:
        response = await gemini.generate_response(prompt)
        
        if isinstance(response, str) and ',' in response:
            parts = response.strip().split(',')
            if len(parts) >= 2:
                topic = parts[0].strip().lower()
                try:
                    confidence = float(parts[1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    return topic, confidence
                except ValueError:
                    pass
    
    except Exception as e:
        print(f"Gemini API error: {e}")
    
    # Fallback
    return "general", 0.5

async def test_topic_detection_performance():
    """Test performance of both methods"""
    print("\n🚀 Performance Test")
    print("=" * 30)
    
    import time
    
    manager = ContextManager()
    test_utterance = "I need directions to the nearest hospital, this is urgent!"
    
    # Test keyword performance
    start_time = time.time()
    for _ in range(100):
        manager.detect_topic_with_confidence(test_utterance)
    keyword_time = time.time() - start_time
    
    print(f"Keyword method: 100 classifications in {keyword_time:.4f}s ({keyword_time*10:.2f}ms each)")
    
    # Test Gemini performance (if available)
    if hasattr(settings, 'GOOGLE_API_KEY') and settings.GOOGLE_API_KEY:
        try:
            gemini = await get_gemini_wrapper()
            await gemini.initialize()
            
            start_time = time.time()
            for _ in range(5):  # Fewer tests due to API limits
                await detect_topic_with_gemini(gemini, test_utterance)
            gemini_time = time.time() - start_time
            
            print(f"Gemini method: 5 classifications in {gemini_time:.4f}s ({gemini_time*200:.2f}ms each)")
            print(f"Speed difference: Keyword is ~{(gemini_time/5)/(keyword_time/100):.0f}x faster")
            
        except Exception as e:
            print(f"Gemini performance test failed: {e}")
    else:
        print("Gemini performance test skipped (no API key)")

async def main():
    """Run all topic detection tests"""
    print("🎯 Topic Detection Comparison Suite")
    print("=" * 60)
    
    tests = [
        test_gemini_vs_keyword_topic_detection,
        test_topic_detection_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if await test():
                passed += 1
                print("\n✅ PASSED")
            else:
                failed += 1
                print("\n❌ FAILED")
        except Exception as e:
            print(f"\n💥 CRASHED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Final Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All topic detection tests completed successfully!")
        print("\n💡 Key Insights:")
        print("- Keyword-based: Fast, reliable for simple cases")
        print("- Gemini-based: Better for complex, nuanced utterances")
        print("- Hybrid approach: Use keywords for speed, Gemini for complexity")
    else:
        print("⚠️ Some tests failed. Check configuration and API keys.")

if __name__ == "__main__":
    asyncio.run(main()) 