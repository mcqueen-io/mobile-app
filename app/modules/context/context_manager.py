from typing import Dict, Optional, List
from datetime import datetime
import json
from app.core.ai_security import get_ai_security_manager
from app.modules.memory.memory_store import get_memory_store
from app.modules.user_info.user_graph import get_user_graph
from app.modules.maps.maps_optimizer import get_maps_optimizer

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

class ContextManager:
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}
        self.ai_security = get_ai_security_manager()
        self.memory_store = get_memory_store()
        self.user_graph = get_user_graph()
        self.maps_optimizer = get_maps_optimizer()
        
    def create_session(self, session_id: str) -> SessionContext:
        """Create a new session context"""
        if session_id in self.sessions:
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
            "timestamp": datetime.utcnow().isoformat()
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
                limit=5
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
    
    def end_session(self, session_id: str) -> None:
        """End a session and clean up resources"""
        if session_id in self.sessions:
            # Save final session data if needed
            session = self.sessions[session_id]
            if session.current_driver:
                # Save session summary to memory
                self.memory_store.add_memory(
                    user_id=session.current_driver,
                    content=f"Session ended at {datetime.utcnow().isoformat()}",
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