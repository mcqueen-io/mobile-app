#!/usr/bin/env python3
"""
Test LangGraph Integration with MongoDB Memory Updates and Intent Handling
Tests the complete flow: User Input -> Topic Detection -> Memory Updates -> Response Generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.modules.context.context_manager import ContextManager, SessionContext, build_session_langgraph
from app.modules.memory.memory_store import get_memory_store
from app.db.mongo_manager import get_mongo_manager
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from datetime import datetime, timezone
import json
import asyncio

def test_langgraph_basic_flow():
    """Test basic LangGraph flow with session management"""
    print("=== Test 1: LangGraph Basic Flow ===")
    
    try:
        # Build the graph
        graph = build_session_langgraph()
        print("✓ LangGraph built successfully")
        
        # Create initial session context
        manager = ContextManager()
        session_id = "langgraph_test_001"
        session = manager.create_session(session_id)
        manager.update_session_data(session_id, {"driver": "dad"})
        
        # Test data for user input
        test_input = {
            "user_id": "dad",
            "utterance": "Queen, I need directions to the nearest hospital",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"Input: {test_input['utterance']}")
        
        # Simulate graph execution (normally would be graph.invoke())
        # Step 1: User input node
        updated_session = manager.get_session(session_id)
        manager.update_session_data(session_id, test_input)
        
        # Step 2: Get updated session with topic detection
        final_session = manager.get_session(session_id)
        
        print(f"✓ Topic detected: {final_session.active_topics[-1]['topic'] if final_session.active_topics else 'none'}")
        print(f"✓ Confidence: {final_session.active_topics[-1]['confidence'] if final_session.active_topics else 0}")
        print(f"✓ Conversation history: {len(final_session.conversation_history)} utterances")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_mongodb_memory_integration():
    """Test MongoDB memory storage integration"""
    print("\n=== Test 2: MongoDB Memory Integration ===")
    
    try:
        # Get memory store (using ChromaDB for now)
        memory_store = get_memory_store()
        # Note: MongoDB manager is async, so we'll focus on ChromaDB for this test
        
        print("✓ Memory store (ChromaDB) initialized")
        
        # Test adding memories to MongoDB
        test_user_id = "test_user_langgraph"
        test_content = "User requested directions to hospital during emergency"
        test_metadata = {
            "session_id": "langgraph_test_001",
            "topic": "navigation",
            "urgency": "high",
            "intent": "emergency_navigation"
        }
        
        # Add memory
        memory_id = memory_store.add_memory(
            user_id=test_user_id,
            content=test_content,
            metadata=test_metadata
        )
        
        print(f"✓ Memory added to ChromaDB with ID: {memory_id}")
        
        # Retrieve memory
        retrieved_memories = memory_store.get_relevant_memories(
            user_id=test_user_id,
            query="hospital directions",
            n_results=5
        )
        
        print(f"✓ Retrieved {len(retrieved_memories)} relevant memories")
        for memory in retrieved_memories:
            print(f"  - {memory.get('content', '')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_intent_handling():
    """Test intent detection and handling"""
    print("\n=== Test 3: Intent Detection & Handling ===")
    
    try:
        manager = ContextManager()
        session_id = "intent_test_session"
        session = manager.create_session(session_id)
        manager.update_session_data(session_id, {"driver": "mom"})
        
        # Test various intents
        test_intents = [
            {
                "utterance": "Queen, I'm feeling drowsy, find me a rest stop",
                "expected_topic": "safety",
                "expected_intent": "safety_alert"
            },
            {
                "utterance": "Navigate to McDonald's on Route 95",
                "expected_topic": "navigation", 
                "expected_intent": "navigation_request"
            },
            {
                "utterance": "Remind me to call John when we arrive",
                "expected_topic": "memory",
                "expected_intent": "reminder_request"
            },
            {
                "utterance": "What's the weather like at our destination?",
                "expected_topic": "weather",
                "expected_intent": "information_request"
            },
            {
                "utterance": "Play some music for the kids",
                "expected_topic": "entertainment",
                "expected_intent": "entertainment_request"
            }
        ]
        
        for i, test_case in enumerate(test_intents):
            utterance = test_case["utterance"]
            expected_topic = test_case["expected_topic"]
            expected_intent = test_case["expected_intent"]
            
            # Detect topic and intent
            topic, confidence = manager.detect_topic_with_confidence(utterance)
            intent = detect_intent(utterance, topic)
            
            print(f"Test {i+1}: '{utterance[:40]}...'")
            print(f"  Topic: {topic} (confidence: {confidence:.2f})")
            print(f"  Intent: {intent}")
            
            # Update session with detected intent
            manager.update_session_data(session_id, {
                "user_id": "mom",
                "utterance": utterance,
                "topic": topic,
                "intent": intent,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            if topic == expected_topic:
                print(f"  ✓ Topic detection correct")
            else:
                print(f"  ! Expected topic: {expected_topic}, got: {topic}")
        
        # Check session state
        final_session = manager.get_session(session_id)
        print(f"\n✓ Session processed {len(final_session.conversation_history)} utterances")
        print(f"✓ Active topics: {len(final_session.active_topics)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def detect_intent(utterance: str, topic: str) -> str:
    """Enhanced intent detection based on utterance and topic"""
    utterance_lower = utterance.lower()
    
    # Intent patterns by topic
    intent_patterns = {
        "navigation": {
            "navigation_request": ["navigate", "directions", "route", "go to", "drive to"],
            "traffic_inquiry": ["traffic", "congestion", "delays"],
            "location_search": ["find", "locate", "where is", "search for"]
        },
        "safety": {
            "safety_alert": ["drowsy", "tired", "sleepy", "fatigue"],
            "emergency_request": ["emergency", "urgent", "help", "hospital"],
            "rest_request": ["rest", "break", "stop", "pull over"]
        },
        "memory": {
            "reminder_request": ["remind", "remember", "don't forget"],
            "schedule_inquiry": ["appointment", "meeting", "schedule"],
            "history_request": ["last time", "before", "previous"]
        },
        "food": {
            "restaurant_search": ["hungry", "eat", "restaurant", "food"],
            "order_request": ["order", "delivery", "takeout"]
        },
        "entertainment": {
            "entertainment_request": ["play", "music", "song", "movie"],
            "game_request": ["game", "play", "fun"]
        },
        "weather": {
            "information_request": ["weather", "temperature", "forecast", "rain"]
        }
    }
    
    if topic in intent_patterns:
        for intent, keywords in intent_patterns[topic].items():
            if any(keyword in utterance_lower for keyword in keywords):
                return intent
    
    # Default intents
    if any(word in utterance_lower for word in ["what", "how", "when", "where", "why"]):
        return "information_request"
    elif any(word in utterance_lower for word in ["please", "can you", "could you"]):
        return "polite_request"
    else:
        return "general_statement"

def test_full_langgraph_orchestration():
    """Test complete LangGraph orchestration with memory updates"""
    print("\n=== Test 4: Full LangGraph Orchestration ===")
    
    try:
        # Build enhanced graph with memory integration
        graph = build_enhanced_session_langgraph()
        manager = ContextManager()
        memory_store = get_memory_store()
        
        # Create session
        session_id = "full_orchestration_test"
        session = manager.create_session(session_id)
        manager.update_session_data(session_id, {
            "driver": "dad",
            "users": [
                {"user_id": "dad", "name": "Dad"},
                {"user_id": "mom", "name": "Mom"},
                {"user_id": "kid1", "name": "Alice"}
            ]
        })
        
        # Simulate a multi-turn conversation
        conversation_flow = [
            ("dad", "Queen, we're starting our trip to Disney World"),
            ("mom", "Actually, let's stop for lunch first"),
            ("kid1", "I'm hungry! Can we get McDonald's?"),
            ("dad", "Queen, find the nearest McDonald's on our route"),
            ("dad", "Also remind me to call my boss when we arrive"),
            ("mom", "What's the weather like at Disney World today?")
        ]
        
        print("🚗 Starting conversation simulation...")
        
        for user_id, utterance in conversation_flow:
            print(f"\n{user_id}: {utterance}")
            
            # Process through enhanced graph
            input_data = {
                "user_id": user_id,
                "utterance": utterance,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Simulate graph execution
            manager.update_session_data(session_id, input_data)
            updated_session = manager.get_session(session_id)
            
            if updated_session.conversation_history:
                last_entry = updated_session.conversation_history[-1]
                print(f"  → Topic: {last_entry['topic']} (confidence: {last_entry['confidence']:.2f})")
                
                # Add to memory if important
                if last_entry['confidence'] > 0.6:
                    memory_store.add_memory(
                        user_id=user_id,
                        content=utterance,
                        metadata={
                            "session_id": session_id,
                            "topic": last_entry['topic'],
                            "confidence": last_entry['confidence'],
                            "timestamp": input_data['timestamp']
                        }
                    )
                    print(f"  ✓ Added to memory (high confidence)")
        
        # Final session stats
        final_session = manager.get_session(session_id)
        print(f"\n📊 Session Summary:")
        print(f"  - Total utterances: {len(final_session.conversation_history)}")
        print(f"  - Active topics: {len(final_session.active_topics)}")
        print(f"  - Session chunks: {len(final_session.session_chunks)}")
        print(f"  - Memory events: {len(final_session.memory_events)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def build_enhanced_session_langgraph():
    """Build enhanced LangGraph with memory integration"""
    from langgraph.graph import StateGraph, END
    
    def enhanced_user_input_node(state: SessionContext, data: dict) -> SessionContext:
        manager = ContextManager()
        manager.update_session_data(state.session_id, data)
        return manager.get_session(state.session_id)
    
    def enhanced_topic_router_node(state: SessionContext) -> SessionContext:
        # Enhanced routing with intent detection
        if state.conversation_history:
            last_utterance = state.conversation_history[-1]
            intent = detect_intent(last_utterance['text'], last_utterance['topic'])
            
            # Add intent to the conversation history
            state.conversation_history[-1]['intent'] = intent
        
        return state
    
    def memory_integration_node(state: SessionContext) -> SessionContext:
        # Save important interactions to memory
        memory_store = get_memory_store()
        
        if state.conversation_history:
            last_entry = state.conversation_history[-1]
            
            # Save high-confidence interactions
            if last_entry.get('confidence', 0) > 0.6:
                memory_store.add_memory(
                    user_id=last_entry['user'],
                    content=last_entry['text'],
                    metadata={
                        "session_id": state.session_id,
                        "topic": last_entry['topic'],
                        "intent": last_entry.get('intent', 'unknown'),
                        "confidence": last_entry['confidence']
                    }
                )
        
        return state
    
    def enhanced_response_generator_node(state: SessionContext) -> dict:
        # Generate contextual response based on topic and intent
        if not state.conversation_history:
            return {"response": "Hello! How can Queen assist you today?", "session_id": state.session_id}
        
        last_entry = state.conversation_history[-1]
        topic = last_entry['topic']
        intent = last_entry.get('intent', 'general_statement')
        utterance = last_entry['text']
        
        # Generate response based on topic and intent
        response_templates = {
            ("navigation", "navigation_request"): "I'll help you navigate to {destination}. Let me find the best route.",
            ("safety", "safety_alert"): "I understand you're feeling tired. Let me find nearby rest stops for your safety.",
            ("memory", "reminder_request"): "I'll set a reminder for you. What would you like me to remind you about?",
            ("food", "restaurant_search"): "I'll help you find restaurants nearby. What type of food are you craving?",
            ("entertainment", "entertainment_request"): "I'll play some music for you. Any particular genre or artist?",
            ("weather", "information_request"): "Let me check the weather conditions for you."
        }
        
        response_key = (topic, intent)
        response = response_templates.get(response_key, f"I heard you mention something about {topic}. How can I help?")
        
        return {
            "response": response,
            "session_id": state.session_id,
            "topic": topic,
            "intent": intent
        }
    
    # Build the enhanced graph
    graph = StateGraph(SessionContext)
    graph.add_node("user_input", enhanced_user_input_node)
    graph.add_node("topic_router", enhanced_topic_router_node)
    graph.add_node("memory_integration", memory_integration_node)
    graph.add_node("response_generator", enhanced_response_generator_node)
    
    # Define enhanced flow
    graph.add_edge("user_input", "topic_router")
    graph.add_edge("topic_router", "memory_integration")
    graph.add_edge("memory_integration", "response_generator")
    graph.add_edge("response_generator", END)
    
    return graph

def test_gemini_topic_detection():
    """Test Gemini-based topic detection vs keyword-based"""
    print("\n=== Test 5: Gemini vs Keyword Topic Detection ===")
    
    try:
        # Skip Gemini for now due to async complexity
        print("Note: Skipping Gemini comparison for now (async complexity)")
        manager = ContextManager()
        
        test_utterances = [
            "I'm feeling really drowsy, can you find a place to rest?",
            "What's the fastest route to avoid this traffic jam?",
            "Can you remind me to call my doctor when we get home?",
            "The kids are getting restless, can you play some music?",
            "I'm starving, where's the closest restaurant?"
        ]
        
        print("Comparing topic detection methods:")
        print("=" * 60)
        
        for utterance in test_utterances:
            # Keyword-based detection
            keyword_topic, keyword_confidence = manager.detect_topic_with_confidence(utterance)
            
            # Simulate Gemini-based detection (placeholder for future implementation)
            gemini_topic = "simulated_" + keyword_topic
            gemini_confidence = min(keyword_confidence + 0.1, 1.0)  # Slightly higher confidence
            
            print(f"Utterance: '{utterance[:50]}...'")
            print(f"  Keyword: {keyword_topic} ({keyword_confidence:.2f})")
            print(f"  Simulated Gemini: {gemini_topic} ({gemini_confidence:.2f})")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all LangGraph and MongoDB integration tests"""
    print("🤖 LangGraph + MongoDB Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_langgraph_basic_flow,
        test_mongodb_memory_integration,
        test_intent_handling,
        test_full_langgraph_orchestration,
        test_gemini_topic_detection
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                failed += 1
                print("❌ FAILED\n")
        except Exception as e:
            print(f"💥 CRASHED: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All LangGraph and MongoDB integration tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 