from typing import Dict, Optional, List, Tuple
from datetime import datetime
import json
import asyncio
from app.core.ai_security import get_ai_security_manager
from app.modules.memory.memory_store import get_memory_store
from app.modules.user_info.user_graph import get_user_graph
from app.modules.maps.maps_optimizer import get_maps_optimizer
from app.modules.ai_wrapper.gemini_wrapper import get_gemini_wrapper
from app.db.mongo_manager import get_mongo_manager
from app.modules.context.conversation_buffer_manager import ConversationBufferManager

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
        # INTELLIGENT CONTEXT FIELDS
        self.conversation_flow: List[Dict] = []  # Full conversation with LLM analysis
        self.important_moments: List[Dict] = []  # LLM-filtered significant events
        self.emotional_context: Dict = {}  # Current emotional state of users
        self.session_themes: List[str] = []  # High-level themes detected by LLM
        self.memory_worthy_events: List[Dict] = []  # Events that should be stored long-term

class IntelligentContextManager:
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}
        self.ai_security = get_ai_security_manager()
        self.memory_store = get_memory_store()
        self.user_graph = get_user_graph()
        self.maps_optimizer = get_maps_optimizer()
        self.gemini = None
        self.mongo_manager = None
        self.conversation_buffer = None
        
    async def initialize(self):
        """Initialize the Gemini wrapper, MongoDB, and conversation buffer for intelligent processing"""
        self.gemini = await get_gemini_wrapper()
        await self.gemini.initialize()
        
        # Initialize MongoDB manager for general memory storage
        from app.core.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        self.mongo_manager = await get_mongo_manager()
        await self.mongo_manager.ensure_initialized()
        
        # Initialize conversation buffer manager
        self.conversation_buffer = ConversationBufferManager(self.gemini, self.memory_store)
        
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
    
    async def process_utterance(self, session_id: str, user_id: str, utterance: str) -> Dict:
        """Process an utterance with conversation buffer management and LLM analysis"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        timestamp = datetime.utcnow()
        
        # Get LLM analysis of the utterance
        analysis = await self._analyze_utterance_with_llm(utterance, user_id, session)
        
        # Add to conversation flow (for immediate context)
        conversation_entry = {
            "user_id": user_id,
            "utterance": utterance,
            "timestamp": timestamp,
            "analysis": analysis
        }
        session.conversation_flow.append(conversation_entry)
        
        # Process with conversation buffer manager
        user_info = session.active_users.get(user_id, {})
        family_tree_id = user_info.get("family_tree_id", "unknown_family")
        
        buffer_result = await self.conversation_buffer.process_utterance(
            session_id, user_id, utterance, analysis, family_tree_id
        )
        
        # Update emotional context
        if analysis.get("emotional_state"):
            session.emotional_context[user_id] = analysis["emotional_state"]
        
        # Check if this is memory-worthy (individual utterance storage)
        memory_stored = False
        if analysis.get("memory_worthy", False) and analysis.get("importance_score", 0) > 0.7:
            memory_event = {
                "user_id": user_id,
                "content": utterance,
                "analysis": analysis,
                "timestamp": timestamp,
                "importance_score": analysis.get("importance_score", 0.5)
            }
            session.memory_worthy_events.append(memory_event)
            await self._store_important_memory(session_id, memory_event)
            memory_stored = True
        
        # Update session themes
        if analysis.get("themes"):
            for theme in analysis["themes"]:
                if theme not in session.session_themes:
                    session.session_themes.append(theme)
        
        # Update last interaction
        session.last_interaction = timestamp
        
        return {
            "analysis": analysis,
            "session_updated": True,
            "memory_stored": memory_stored,
            "buffer_info": buffer_result,
            "topic_shift": buffer_result.get("topic_shifted", False),
            "current_topic": buffer_result.get("topic", "general"),
            "chunk_size": buffer_result.get("chunk_size", 0)
        }
    
    async def _analyze_utterance_with_llm(self, utterance: str, user_id: str, session: SessionContext) -> Dict:
        """Use Gemini to analyze utterance for topics, emotions, importance, and intent"""
        
        # Build context for the LLM
        context_info = {
            "recent_conversation": session.conversation_flow[-5:] if session.conversation_flow else [],
            "active_users": list(session.active_users.keys()),
            "current_emotional_context": session.emotional_context,
            "session_themes": session.session_themes,
            "trip_context": {
                "destination": session.destination,
                "current_location": session.current_location
            }
        }
        
        prompt = f"""
You are Queen, an intelligent in-car AI assistant. Analyze this utterance in context:

UTTERANCE: "{utterance}"
USER: {user_id}
CONTEXT: {json.dumps(context_info, indent=2, default=str)}

Provide a JSON analysis with these fields:

{{
    "topic": "primary topic (navigation, safety, entertainment, food, memory, weather, personal, general)",
    "intent": "specific intent (request_navigation, express_emotion, make_plan, ask_question, etc.)",
    "emotional_state": "user's emotional state (happy, frustrated, tired, excited, neutral, etc.)",
    "importance_score": 0.0-1.0,
    "memory_worthy": true/false,
    "themes": ["list", "of", "conversation", "themes"],
    "requires_response": true/false,
    "urgency": "low/medium/high",
    "reasoning": "why you classified it this way"
}}

IMPORTANCE SCORING GUIDELINES:
- 0.9-1.0: Life events, emergencies, important appointments, strong emotions
- 0.7-0.8: Plans, preferences, meaningful conversations, safety concerns  
- 0.5-0.6: Casual requests, mild preferences, routine questions
- 0.3-0.4: Small talk, routine tasks, simple confirmations
- 0.0-0.2: Mundane chatter, "going to McDonald's", weather small talk

MEMORY WORTHY CRITERIA:
- Personal revelations or strong emotions
- Important plans or appointments  
- Safety concerns or health issues
- Relationship dynamics or family moments
- NOT routine tasks like "let's get food" unless emotionally significant

Respond with ONLY the JSON object:
"""
        
        try:
            response = await self.gemini.generate_response(prompt)
            
            # Parse JSON response
            if isinstance(response, str):
                # Clean up response to extract JSON
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:-3]
                elif response.startswith('```'):
                    response = response[3:-3]
                
                analysis = json.loads(response)
                
                # Validate and clean the analysis
                analysis = self._validate_analysis(analysis)
                return analysis
                
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            # Fallback to basic analysis
            return self._fallback_analysis(utterance)
    
    def _validate_analysis(self, analysis: Dict) -> Dict:
        """Validate and clean LLM analysis response"""
        validated = {
            "topic": analysis.get("topic", "general"),
            "intent": analysis.get("intent", "general_statement"),
            "emotional_state": analysis.get("emotional_state", "neutral"),
            "importance_score": max(0.0, min(1.0, float(analysis.get("importance_score", 0.5)))),
            "memory_worthy": bool(analysis.get("memory_worthy", False)),
            "themes": analysis.get("themes", []),
            "requires_response": bool(analysis.get("requires_response", True)),
            "urgency": analysis.get("urgency", "low"),
            "reasoning": analysis.get("reasoning", "No reasoning provided")
        }
        
        # Ensure memory_worthy aligns with importance_score
        if validated["importance_score"] > 0.7:
            validated["memory_worthy"] = True
        elif validated["importance_score"] < 0.4:
            validated["memory_worthy"] = False
            
        return validated
    
    def _fallback_analysis(self, utterance: str) -> Dict:
        """Fallback analysis when LLM fails"""
        return {
            "topic": "general",
            "intent": "general_statement",
            "emotional_state": "neutral",
            "importance_score": 0.3,
            "memory_worthy": False,
            "themes": [],
            "requires_response": True,
            "urgency": "low",
            "reasoning": "Fallback analysis - LLM unavailable"
        }
    
    async def _store_important_memory(self, session_id: str, memory_event: Dict):
        """Store chat memories in ChromaDB for semantic search, general memories in MongoDB"""
        try:
            memory_content = f"User {memory_event['user_id']}: {memory_event['content']}"
            
            # Get family_tree_id from session context
            session = self.get_session(session_id)
            user_info = session.active_users.get(memory_event["user_id"], {}) if session else {}
            family_tree_id = user_info.get("family_tree_id", "unknown_family")
            
            # Generate conversation_id for linking related messages
            conversation_id = f"{session_id}_{memory_event['timestamp'].strftime('%Y%m%d_%H%M')}"
            
            # ChromaDB metadata (for semantic vector search of conversations)
            chroma_metadata = {
                "session_id": session_id,
                "conversation_id": conversation_id,
                "family_tree_id": family_tree_id,
                "user_id": memory_event["user_id"],
                "topic": memory_event["analysis"]["topic"],
                "emotional_state": memory_event["analysis"]["emotional_state"],
                "importance_score": memory_event["analysis"]["importance_score"],
                "themes": ",".join(memory_event["analysis"]["themes"]) if memory_event["analysis"]["themes"] else "",
                "timestamp": memory_event["timestamp"].isoformat(),
                "reasoning": memory_event["analysis"]["reasoning"]
            }
            
            # Store conversation semantics in ChromaDB for vector search
            chroma_memory_id = self.memory_store.add_memory(
                user_id=memory_event["user_id"],
                content=memory_content,
                metadata=chroma_metadata
            )
            
            print(f"✓ Stored conversation semantics in ChromaDB: {chroma_memory_id}")
            
            # For now, we only store chat semantics in ChromaDB
            # MongoDB is reserved for structured user data, preferences, family trees, etc.
            # Not for individual chat memories
            
            return {
                "chroma_id": chroma_memory_id
            }
            
        except Exception as e:
            print(f"Failed to store memory: {e}")
            return None
    
    async def get_intelligent_context(self, session_id: str, query: str) -> Dict:
        """Get intelligent context for response generation"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Analyze the query to understand what context is needed
        query_analysis = await self._analyze_utterance_with_llm(query, "system", session)
        
        context = {
            "session_id": session_id,
            "current_emotional_context": session.emotional_context,
            "session_themes": session.session_themes,
            "recent_important_moments": session.memory_worthy_events[-3:],
            "query_analysis": query_analysis,
            "conversation_flow": session.conversation_flow[-5:],  # Recent context
        }
        
        # Get relevant memories based on query
        if session.current_driver:
            try:
                relevant_memories = self.memory_store.get_relevant_memories(
                    user_id=session.current_driver,
                    query=query,
                    n_results=3
                )
                context["relevant_memories"] = relevant_memories
            except Exception as e:
                print(f"Memory retrieval failed: {e}")
                context["relevant_memories"] = []
        
        return context
    
    async def generate_intelligent_response(self, session_id: str, query: str, user_id: str) -> str:
        """Generate contextually intelligent response using Queen's personality"""
        
        # First, process the query as an utterance
        await self.process_utterance(session_id, user_id, query)
        
        # Get intelligent context
        context = await self.get_intelligent_context(session_id, query)
        
        # Build response prompt for Queen
        response_prompt = f"""
You are Queen, an empathetic and intelligent in-car AI assistant. Respond to this user naturally and helpfully.

USER QUERY: "{query}"
USER ID: {user_id}

CONTEXT:
{json.dumps(context, indent=2, default=str)}

RESPONSE GUIDELINES:
- Be warm, helpful, and contextually aware
- Reference emotional states and previous conversation naturally
- Provide actionable assistance for navigation, safety, entertainment, etc.
- Keep responses concise but thoughtful
- Address the user by name when appropriate
- Show empathy for emotions and concerns

Generate a natural, helpful response as Queen:
"""
        
        try:
            response = await self.gemini.generate_response(response_prompt)
            return response.strip() if isinstance(response, str) else "I'm here to help! How can I assist you?"
            
        except Exception as e:
            print(f"Response generation failed: {e}")
            return "I understand what you're saying. How can I help you with that?"
    
    async def end_session(self, session_id: str) -> Dict:
        """End a session and complete all conversation chunks"""
        if self.conversation_buffer:
            await self.conversation_buffer.end_session(session_id)
        
        session_summary = self.get_session_summary(session_id)
        
        # Remove session from active sessions
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        return session_summary
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get intelligent summary of session including conversation chunks"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        base_summary = {
            "session_id": session_id,
            "duration": (datetime.utcnow() - session.start_time).total_seconds(),
            "total_utterances": len(session.conversation_flow),
            "memory_worthy_events": len(session.memory_worthy_events),
            "session_themes": session.session_themes,
            "emotional_context": session.emotional_context,
            "important_moments": len([e for e in session.memory_worthy_events if e["importance_score"] > 0.7])
        }
        
        # Add conversation buffer summary if available
        if self.conversation_buffer:
            buffer_summary = self.conversation_buffer.get_session_summary(session_id)
            base_summary.update({
                "conversation_chunks": buffer_summary.get("total_chunks", 0),
                "chunk_topics": buffer_summary.get("topics", []),
                "chunk_details": buffer_summary.get("chunks", [])
            })
        
        return base_summary

# Create singleton instance
intelligent_context_manager = IntelligentContextManager()

async def get_intelligent_context_manager():
    if intelligent_context_manager.gemini is None:
        await intelligent_context_manager.initialize()
    return intelligent_context_manager 