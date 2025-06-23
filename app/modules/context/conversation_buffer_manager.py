#!/usr/bin/env python3
"""
Conversation Buffer Manager
Handles topic-based conversation chunking, summarization, and intelligent storage
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import json

@dataclass
class ConversationChunk:
    """A chunk of conversation focused on a specific topic"""
    chunk_id: str
    session_id: str
    family_tree_id: str
    topic: str
    start_time: datetime
    end_time: datetime
    participants: List[str]
    utterances: List[Dict]
    summary: str
    key_points: List[str]
    emotional_tone: str
    importance_score: float
    themes: List[str]

class ConversationBufferManager:
    """Manages conversation chunks with topic detection and summarization"""
    
    def __init__(self, gemini_wrapper, memory_store):
        self.gemini = gemini_wrapper
        self.memory_store = memory_store
        self.active_chunks: Dict[str, ConversationChunk] = {}  # session_id -> current chunk
        self.chunk_history: Dict[str, List[ConversationChunk]] = {}  # session_id -> completed chunks
        
        # Configuration
        self.max_chunk_utterances = 15  # Max utterances before forcing chunk
        self.topic_shift_threshold = 0.7  # Similarity threshold for topic detection
        self.chunk_timeout_minutes = 10  # Auto-chunk after silence
        
    async def process_utterance(self, session_id: str, user_id: str, utterance: str, 
                               analysis: Dict, family_tree_id: str) -> Dict:
        """Process an utterance and manage conversation chunks"""
        
        timestamp = datetime.now(timezone.utc)
        utterance_data = {
            "user_id": user_id,
            "utterance": utterance,
            "analysis": analysis,
            "timestamp": timestamp
        }
        
        # Check if we need to start a new chunk or continue existing
        current_chunk = self.active_chunks.get(session_id)
        
        if not current_chunk:
            # Start new chunk
            current_chunk = await self._start_new_chunk(
                session_id, family_tree_id, utterance_data
            )
        else:
            # Check for topic shift
            topic_shifted = await self._detect_topic_shift(current_chunk, utterance_data)
            
            if topic_shifted or self._should_force_chunk(current_chunk):
                # Complete current chunk and start new one
                await self._complete_chunk(session_id, current_chunk)
                current_chunk = await self._start_new_chunk(
                    session_id, family_tree_id, utterance_data
                )
            else:
                # Add to current chunk
                self._add_to_chunk(current_chunk, utterance_data)
        
        self.active_chunks[session_id] = current_chunk
        
        return {
            "chunk_id": current_chunk.chunk_id,
            "topic": current_chunk.topic,
            "chunk_size": len(current_chunk.utterances),
            "topic_shifted": topic_shifted if current_chunk else False
        }
    
    async def _start_new_chunk(self, session_id: str, family_tree_id: str, 
                              utterance_data: Dict) -> ConversationChunk:
        """Start a new conversation chunk"""
        
        chunk_id = f"{session_id}_{utterance_data['timestamp'].strftime('%Y%m%d_%H%M%S')}"
        topic = utterance_data["analysis"].get("topic", "general")
        
        chunk = ConversationChunk(
            chunk_id=chunk_id,
            session_id=session_id,
            family_tree_id=family_tree_id,
            topic=topic,
            start_time=utterance_data["timestamp"],
            end_time=utterance_data["timestamp"],
            participants=[utterance_data["user_id"]],
            utterances=[utterance_data],
            summary="",
            key_points=[],
            emotional_tone=utterance_data["analysis"].get("emotional_state", "neutral"),
            importance_score=utterance_data["analysis"].get("importance_score", 0.5),
            themes=utterance_data["analysis"].get("themes", [])
        )
        
        return chunk
    
    def _add_to_chunk(self, chunk: ConversationChunk, utterance_data: Dict):
        """Add utterance to existing chunk"""
        
        chunk.utterances.append(utterance_data)
        chunk.end_time = utterance_data["timestamp"]
        
        # Update participants
        user_id = utterance_data["user_id"]
        if user_id not in chunk.participants:
            chunk.participants.append(user_id)
        
        # Update themes
        new_themes = utterance_data["analysis"].get("themes", [])
        for theme in new_themes:
            if theme not in chunk.themes:
                chunk.themes.append(theme)
        
        # Update importance (take maximum)
        new_importance = utterance_data["analysis"].get("importance_score", 0.5)
        chunk.importance_score = max(chunk.importance_score, new_importance)
    
    async def _detect_topic_shift(self, current_chunk: ConversationChunk, 
                                 utterance_data: Dict) -> bool:
        """Detect if there's a significant topic shift"""
        
        current_topic = current_chunk.topic
        new_topic = utterance_data["analysis"].get("topic", "general")
        
        # Simple topic comparison (can be enhanced with semantic similarity)
        if current_topic != new_topic:
            # Use LLM to determine if this is a significant shift
            return await self._llm_topic_shift_analysis(current_chunk, utterance_data)
        
        return False
    
    async def _llm_topic_shift_analysis(self, current_chunk: ConversationChunk, 
                                       utterance_data: Dict) -> bool:
        """Use LLM to analyze if topic shift is significant enough to chunk"""
        
        # Get recent utterances from current chunk
        recent_utterances = current_chunk.utterances[-3:]
        recent_text = " ".join([u["utterance"] for u in recent_utterances])
        
        new_utterance = utterance_data["utterance"]
        
        prompt = f"""
Analyze if this represents a significant topic shift requiring a new conversation chunk:

CURRENT CHUNK TOPIC: {current_chunk.topic}
RECENT CONVERSATION: "{recent_text}"
NEW UTTERANCE: "{new_utterance}"

Consider:
- Are they discussing the same subject matter?
- Is this a natural continuation or a complete change?
- Would breaking here preserve conversational context?

Respond with JSON:
{{
    "topic_shift": true/false,
    "reasoning": "explanation of decision"
}}
"""
        
        try:
            response = await self.gemini.generate_response(prompt)
            
            # Parse response
            if isinstance(response, str):
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:-3]
                elif response.startswith('```'):
                    response = response[3:-3]
                
                result = json.loads(response)
                return result.get("topic_shift", False)
                
        except Exception as e:
            print(f"Topic shift analysis failed: {e}")
            # Fallback: different topics = shift
            return current_chunk.topic != utterance_data["analysis"].get("topic", "general")
        
        return False
    
    def _should_force_chunk(self, chunk: ConversationChunk) -> bool:
        """Check if chunk should be forced due to size or time"""
        
        # Force chunk if too many utterances
        if len(chunk.utterances) >= self.max_chunk_utterances:
            return True
        
        # Force chunk if too much time has passed
        time_diff = datetime.now(timezone.utc) - chunk.end_time
        if time_diff.total_seconds() > (self.chunk_timeout_minutes * 60):
            return True
        
        return False
    
    async def _complete_chunk(self, session_id: str, chunk: ConversationChunk):
        """Complete a chunk by generating summary and storing"""
        
        # Generate summary
        chunk.summary = await self._generate_chunk_summary(chunk)
        chunk.key_points = await self._extract_key_points(chunk)
        
        # Store chunk summary in long-term memory
        await self._store_chunk_summary(chunk)
        
        # Add to history
        if session_id not in self.chunk_history:
            self.chunk_history[session_id] = []
        self.chunk_history[session_id].append(chunk)
        
        print(f"✓ Completed conversation chunk: {chunk.chunk_id}")
        print(f"  Topic: {chunk.topic}")
        print(f"  Utterances: {len(chunk.utterances)}")
        print(f"  Participants: {chunk.participants}")
        print(f"  Summary: {chunk.summary[:100]}...")
    
    async def _generate_chunk_summary(self, chunk: ConversationChunk) -> str:
        """Generate a concise summary of the conversation chunk"""
        
        # Build conversation text
        conversation_text = []
        for utterance in chunk.utterances:
            user_id = utterance["user_id"]
            text = utterance["utterance"]
            conversation_text.append(f"{user_id}: {text}")
        
        full_conversation = "\n".join(conversation_text)
        
        prompt = f"""
Summarize this conversation chunk concisely:

TOPIC: {chunk.topic}
PARTICIPANTS: {', '.join(chunk.participants)}
CONVERSATION:
{full_conversation}

Create a 2-3 sentence summary that captures:
- Main topic/subject
- Key decisions or outcomes
- Important emotional context
- Any action items or plans

Summary:"""
        
        try:
            summary = await self.gemini.generate_response(prompt)
            return summary.strip() if isinstance(summary, str) else "Conversation summary unavailable"
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return f"Conversation about {chunk.topic} with {len(chunk.utterances)} messages"
    
    async def _extract_key_points(self, chunk: ConversationChunk) -> List[str]:
        """Extract key points from the conversation chunk"""
        
        conversation_text = []
        for utterance in chunk.utterances:
            user_id = utterance["user_id"]
            text = utterance["utterance"]
            conversation_text.append(f"{user_id}: {text}")
        
        full_conversation = "\n".join(conversation_text)
        
        prompt = f"""
Extract 3-5 key points from this conversation:

{full_conversation}

Return as JSON array:
["key point 1", "key point 2", "key point 3"]

Focus on:
- Important decisions
- Emotional moments
- Plans or commitments
- Safety concerns
- Family dynamics
"""
        
        try:
            response = await self.gemini.generate_response(prompt)
            
            if isinstance(response, str):
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:-3]
                elif response.startswith('```'):
                    response = response[3:-3]
                
                key_points = json.loads(response)
                return key_points if isinstance(key_points, list) else []
                
        except Exception as e:
            print(f"Key point extraction failed: {e}")
            return []
        
        return []
    
    async def _store_chunk_summary(self, chunk: ConversationChunk):
        """Store chunk summary in ChromaDB for long-term retrieval"""
        
        # Create summary content for storage
        summary_content = f"""
CONVERSATION SUMMARY
Topic: {chunk.topic}
Participants: {', '.join(chunk.participants)}
Duration: {chunk.start_time.strftime('%H:%M')} - {chunk.end_time.strftime('%H:%M')}

Summary: {chunk.summary}

Key Points:
{chr(10).join(f"• {point}" for point in chunk.key_points)}

Themes: {', '.join(chunk.themes)}
"""
        
        # Metadata for the summary
        summary_metadata = {
            "type": "conversation_summary",
            "chunk_id": chunk.chunk_id,
            "session_id": chunk.session_id,
            "family_tree_id": chunk.family_tree_id,
            "topic": chunk.topic,
            "participants": ",".join(chunk.participants),
            "start_time": chunk.start_time.isoformat(),
            "end_time": chunk.end_time.isoformat(),
            "utterance_count": len(chunk.utterances),
            "importance_score": chunk.importance_score,
            "emotional_tone": chunk.emotional_tone,
            "themes": ",".join(chunk.themes),
            "timestamp": chunk.end_time.isoformat()
        }
        
        # Store in ChromaDB
        summary_id = self.memory_store.add_memory(
            user_id=f"summary_{chunk.session_id}",
            content=summary_content,
            metadata=summary_metadata
        )
        
        print(f"✓ Stored chunk summary: {summary_id}")
    
    async def get_topic_context(self, session_id: str, topic: str, max_chunks: int = 3) -> List[ConversationChunk]:
        """Get previous conversation chunks about a specific topic"""
        
        session_chunks = self.chunk_history.get(session_id, [])
        
        # Find chunks with matching or related topics
        relevant_chunks = []
        for chunk in reversed(session_chunks):  # Most recent first
            if chunk.topic == topic or topic in chunk.themes:
                relevant_chunks.append(chunk)
                if len(relevant_chunks) >= max_chunks:
                    break
        
        return relevant_chunks
    
    async def end_session(self, session_id: str):
        """End session and complete any active chunks"""
        
        if session_id in self.active_chunks:
            chunk = self.active_chunks[session_id]
            await self._complete_chunk(session_id, chunk)
            del self.active_chunks[session_id]
        
        print(f"✓ Session {session_id} ended, all chunks completed")
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of all chunks in a session"""
        
        chunks = self.chunk_history.get(session_id, [])
        active_chunk = self.active_chunks.get(session_id)
        
        if active_chunk:
            chunks = chunks + [active_chunk]
        
        if not chunks:
            return {"total_chunks": 0, "topics": [], "participants": []}
        
        # Aggregate data
        all_topics = []
        all_participants = set()
        total_utterances = 0
        
        for chunk in chunks:
            all_topics.append(chunk.topic)
            all_participants.update(chunk.participants)
            total_utterances += len(chunk.utterances)
        
        return {
            "total_chunks": len(chunks),
            "topics": list(set(all_topics)),
            "participants": list(all_participants),
            "total_utterances": total_utterances,
            "session_duration": (chunks[-1].end_time - chunks[0].start_time).total_seconds() / 60,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "topic": chunk.topic,
                    "summary": chunk.summary,
                    "importance": chunk.importance_score
                }
                for chunk in chunks
            ]
        } 