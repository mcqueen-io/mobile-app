from typing import Dict, Optional, List
from datetime import datetime
import json
from app.core.ai_security import get_ai_security_manager
from app.modules.memory.memory_store import get_memory_store
from app.modules.user_info.user_graph import get_user_graph
from app.modules.maps.maps_optimizer import get_maps_optimizer
from langgraph.graph import StateGraph, END

class SessionContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.utcnow()
        self.active_users: Dict[str, Dict] = {}  # user_id -> user_data
        self.current_driver: Optional[str] = None
        self.current_location: Optional[Dict] = None
        self.destination: Optional[Dict] = None
        self.current_route: Optional[Dict] = None
        self.last_interaction: Optional[datetime] = None
        self.interaction_history: List[Dict] = []
        self.required_context: Dict[str, bool] = {
            "user_info": False,
            "memory": False,
            "maps": False,
            "voice": False
        }
        # ENHANCED FIELDS FOR MULTI-USER, MULTI-TOPIC, LANGGRAPH
        self.active_topics: List[Dict] = []  # [{topic, start_time, context_chunk, end_time}]
        self.conversation_history: List[Dict] = []  # [{user, text, timestamp, topic}]
        self.memory_events: List[Dict] = []  # [{event_type, data, timestamp}]
        self.session_chunks: List[Dict] = []  # [{start, end, topic, chunk_data}]

class ContextManager:
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}
        self.ai_security = get_ai_security_manager()
        self.memory_store = get_memory_store()
        self.user_graph = get_user_graph()
        self.maps_optimizer = get_maps_optimizer()
        
    def create_session(self, session_id: str, allow_existing: bool = False) -> SessionContext:
        """Create a new session context"""
        if session_id in self.sessions:
            if allow_existing:
                return self.sessions[session_id]
            else:
                raise ValueError(f"Session {session_id} already exists")
        
        session = SessionContext(session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get an existing session context"""
        return self.sessions.get(session_id)
    
    def update_session_data(self, session_id: str, data: Dict) -> None:
        """Update session data with new information"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Update active users
        if "users" in data:
            for user_data in data["users"]:
                user_id = user_data.get("user_id")
                if user_id:
                    session.active_users[user_id] = user_data
        
        # Update driver information
        if "driver" in data:
            session.current_driver = data["driver"]
        
        # Update location information
        if "location" in data:
            session.current_location = data["location"]
        
        # Update destination
        if "destination" in data:
            session.destination = data["destination"]
            # Update route if destination changes
            if session.current_location:
                self._update_route(session)
        
        # Update last interaction time
        session.last_interaction = datetime.utcnow()
        
        # Add to interaction history
        session.interaction_history.append({
            "timestamp": session.last_interaction,
            "data": data
        })
        
        # --- ENHANCED: Handle utterance/topic tracking with improved chunking ---
        if "utterance" in data and "user_id" in data:
            utterance = data["utterance"]
            user_id = data["user_id"]
            timestamp = datetime.utcnow()
            topic, confidence = self.detect_topic_with_confidence(utterance)
            
            # Check if we need to create a new topic chunk
            should_chunk = False
            
            if not session.active_topics:
                # First topic
                should_chunk = True
            else:
                last_topic = session.active_topics[-1]
                # Chunk if topic changed with high confidence or if chunk is getting too large
                if (last_topic["topic"] != topic and confidence > 0.7) or len(last_topic["context_chunk"]) >= 10:
                    should_chunk = True
            
            if should_chunk:
                # Close previous topic if exists
                if session.active_topics:
                    session.active_topics[-1]["end_time"] = timestamp
                    session.session_chunks.append({
                        "start": session.active_topics[-1]["start_time"],
                        "end": timestamp,
                        "topic": session.active_topics[-1]["topic"],
                        "chunk_data": session.active_topics[-1]["context_chunk"],
                        "utterance_count": len(session.active_topics[-1]["context_chunk"])
                    })
                
                # Start new topic
                session.active_topics.append({
                    "topic": topic,
                    "confidence": confidence,
                    "start_time": timestamp,
                    "context_chunk": []
                })
            
            # Add utterance to current topic
            session.active_topics[-1]["context_chunk"].append({
                "user": user_id,
                "text": utterance,
                "timestamp": timestamp
            })
            
            session.conversation_history.append({
                "user": user_id,
                "text": utterance,
                "timestamp": timestamp,
                "topic": topic,
                "confidence": confidence
            })
    
    def get_context_for_query(self, session_id: str, query: str) -> Dict:
        """Get relevant context for a specific query"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        context = {
            "session_id": session_id,
            "current_driver": session.current_driver,
            "current_location": session.current_location,
            "destination": session.destination,
            "timestamp": datetime.utcnow().isoformat(),
            "active_topics": session.active_topics,
            "conversation_history": session.conversation_history,
            "memory_events": session.memory_events,
            "session_chunks": session.session_chunks
        }
        
        # Determine which modules' data is needed
        self._determine_required_context(query, session)
        
        # Get user information if needed
        if session.required_context["user_info"] and session.current_driver:
            user_data = self.user_graph.get_user_preferences(session.current_driver)
            context["user_preferences"] = user_data
        
        # Get relevant memories if needed
        if session.required_context["memory"]:
            memories = self.memory_store.get_relevant_memories(
                session.current_driver,
                query,
                n_results=5
            )
            context["relevant_memories"] = memories
        
        # Get route information if needed
        if session.required_context["maps"]:
            if session.current_route:
                context["current_route"] = session.current_route
            else:
                self._update_route(session)
                context["current_route"] = session.current_route
        
        return context
    
    def _determine_required_context(self, query: str, session: SessionContext) -> None:
        """Determine which modules' data is needed for the query"""
        # Reset required context
        session.required_context = {k: False for k in session.required_context}
        
        # Check for user-related queries
        user_keywords = ["my", "I", "me", "user", "preference", "like", "dislike"]
        if any(keyword in query.lower() for keyword in user_keywords):
            session.required_context["user_info"] = True
        
        # Check for memory-related queries
        memory_keywords = ["remember", "before", "last time", "previous", "history"]
        if any(keyword in query.lower() for keyword in memory_keywords):
            session.required_context["memory"] = True
        
        # Check for navigation-related queries
        nav_keywords = ["where", "route", "direction", "navigate", "go to", "drive"]
        if any(keyword in query.lower() for keyword in nav_keywords):
            session.required_context["maps"] = True
    
    def _update_route(self, session: SessionContext) -> None:
        """Update the current route based on location and destination"""
        if session.current_location and session.destination:
            try:
                route_data = self.maps_optimizer.optimize_route(
                    start_location=session.current_location,
                    end_location=session.destination
                )
                session.current_route = route_data
            except Exception as e:
                print(f"Error updating route: {str(e)}")
    
    def detect_topic(self, utterance: str) -> str:
        """Simple topic detection (backward compatibility)"""
        topic, _ = self.detect_topic_with_confidence(utterance)
        return topic
    
    def detect_topic_with_confidence(self, utterance: str) -> tuple[str, float]:
        """Enhanced topic detection with confidence scoring"""
        utterance_lower = utterance.lower()
        
        # Define topic keywords with weights
        topic_keywords = {
            "navigation": {
                "keywords": ["route", "navigate", "direction", "drive", "traffic", "stop", "destination", "gps", "map"],
                "weight": 1.0
            },
            "memory": {
                "keywords": ["remember", "remind", "event", "history", "appointment", "schedule", "calendar"],
                "weight": 1.0
            },
            "safety": {
                "keywords": ["tired", "sleep", "rest", "break", "fatigue", "drowsy", "alert"],
                "weight": 1.0
            },
            "food": {
                "keywords": ["hungry", "eat", "food", "restaurant", "mcdonald", "lunch", "dinner", "snack"],
                "weight": 0.8
            },
            "weather": {
                "keywords": ["weather", "rain", "sunny", "cloudy", "temperature", "forecast"],
                "weight": 0.8
            },
            "entertainment": {
                "keywords": ["music", "play", "song", "movie", "game", "fun", "boring"],
                "weight": 0.7
            }
        }
        
        topic_scores = {}
        
        for topic, config in topic_keywords.items():
            score = 0
            for keyword in config["keywords"]:
                if keyword in utterance_lower:
                    score += config["weight"]
            
            if score > 0:
                # Normalize by utterance length to avoid bias toward long utterances
                normalized_score = min(score / max(len(utterance_lower.split()), 1), 1.0)
                topic_scores[topic] = normalized_score
        
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            confidence = topic_scores[best_topic]
            return best_topic, confidence
        else:
            return "general", 0.5

    def log_memory_event(self, session_id: str, event_type: str, data: Dict):
        session = self.get_session(session_id)
        if session:
            session.memory_events.append({
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.utcnow()
            })
    
    def end_session(self, session_id: str) -> None:
        """End a session and clean up resources"""
        if session_id in self.sessions:
            # Save final session data if needed
            session = self.sessions[session_id]
            if session.current_driver:
                # Save session summary to memory
                self.memory_store.add_memory(
                    user_id=session.current_driver or "unknown",
                    content=json.dumps({
                        "session_id": session_id,
                        "start_time": session.start_time.isoformat(),
                        "end_time": datetime.utcnow().isoformat(),
                        "topics": session.session_chunks,
                        "memory_events": session.memory_events,
                        "conversation_history": session.conversation_history
                    }),
                    metadata={
                        "session_id": session_id,
                        "duration": (datetime.utcnow() - session.start_time).total_seconds(),
                        "interaction_count": len(session.interaction_history)
                    }
                )
            del self.sessions[session_id]

# Create singleton instance
context_manager = ContextManager()

def get_context_manager():
    return context_manager 

# --- LangGraph Integration ---

def build_session_langgraph():
    """
    Build a LangGraph for session orchestration using SessionContext as state.
    Nodes: user_input, topic_router, response_generator (expandable).
    """
    def user_input_node(state: SessionContext, data: dict) -> SessionContext:
        # Simulate user utterance input (data should have 'utterance' and 'user_id')
        # This will use the enhanced update_session_data logic
        context_manager.update_session_data(state.session_id, data)
        return context_manager.get_session(state.session_id)

    def topic_router_node(state: SessionContext) -> SessionContext:
        # In a real system, this could do more advanced topic detection/routing
        # Here, we just return the state (topic detection is in update_session_data)
        return state

    def response_generator_node(state: SessionContext) -> dict:
        # Generate a response for the current session state (stub)
        # In a real system, this would call the LLM/Queen
        last_utterance = state.conversation_history[-1]["text"] if state.conversation_history else ""
        response = f"Queen heard: '{last_utterance}' (topic: {state.active_topics[-1]['topic'] if state.active_topics else 'general'})"
        return {"response": response, "session_id": state.session_id}

    # Build the graph
    graph = StateGraph(SessionContext)
    graph.add_node("user_input", user_input_node)
    graph.add_node("topic_router", topic_router_node)
    graph.add_node("response_generator", response_generator_node)
    # Define flow: user_input -> topic_router -> response_generator -> END
    graph.add_edge("user_input", "topic_router")
    graph.add_edge("topic_router", "response_generator")
    graph.add_edge("response_generator", END)
    return graph 