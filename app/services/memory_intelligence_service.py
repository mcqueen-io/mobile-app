import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import re
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

from app.db.mongo_manager import get_mongo_manager
from app.modules.memory.memory_store import get_memory_store
from app.core.config import settings

logger = logging.getLogger(__name__)

class MemoryIntelligenceService:
    """Service for intelligent memory formation and event extraction from conversations"""
    
    def __init__(self):
        self.mongo_manager = None
        self.memory_store = get_memory_store()
        
    async def initialize(self):
        """Initialize async dependencies"""
        self.mongo_manager = await get_mongo_manager()
        
    async def extract_and_store_events(self, conversation_text: str, participants: List[str]) -> Dict[str, Any]:
        """
        Extract important events from conversation and store them with reflection triggers.
        This is called by the Gemini wrapper when it detects significant events.
        """
        try:
            if not self.mongo_manager:
                await self.initialize()
                
            # Extract structured events from conversation
            extracted_events = await self._analyze_conversation_for_events(conversation_text, participants)
            
            if not extracted_events:
                return {
                    "success": True,
                    "events_found": 0,
                    "message": "No significant events detected in conversation"
                }
            
            # Store each extracted event
            stored_events = []
            for event in extracted_events:
                try:
                    stored_event = await self._store_intelligent_memory(event, participants)
                    if stored_event:
                        stored_events.append(stored_event)
                except Exception as e:
                    logger.error(f"Error storing event {event}: {str(e)}")
                    continue
            
            return {
                "success": True,
                "events_found": len(extracted_events),
                "events_stored": len(stored_events),
                "stored_events": stored_events,
                "message": f"Successfully extracted and stored {len(stored_events)} events for future reflection"
            }
            
        except Exception as e:
            logger.error(f"Error in extract_and_store_events: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "events_found": 0
            }
    
    async def _analyze_conversation_for_events(self, conversation_text: str, participants: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze conversation text and extract structured event data.
        Uses pattern matching and NLP techniques to identify important events.
        """
        events = []
        
        # Define event patterns and extraction rules
        event_patterns = [
            # Interview patterns
            {
                "pattern": r"(?i)(interview|job interview).{0,50}(?:at|with)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+(?:on|next|this)\s+([a-zA-Z]+)|(?:\s+for\s+([^.!?]+)))?",
                "event_type": "interview",
                "extract_fields": ["company", "date", "role"]
            },
            # Meeting patterns
            {
                "pattern": r"(?i)(meeting|appointment).{0,30}(?:with|at)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+(?:on|next|this)\s+([a-zA-Z]+))?",
                "event_type": "meeting",
                "extract_fields": ["company", "date"]
            },
            # Birthday patterns
            {
                "pattern": r"(?i)([A-Z][a-zA-Z]+)'?s?\s+birthday.{0,20}(?:on|next|this)\s+([a-zA-Z\s]+)",
                "event_type": "birthday",
                "extract_fields": ["person", "date"]
            },
            # Trip patterns
            {
                "pattern": r"(?i)(trip|vacation|travel).{0,30}(?:to|in)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:on|next|this|in)\s+([a-zA-Z\s]+))?",
                "event_type": "trip",
                "extract_fields": ["destination", "date"]
            },
            # Presentation patterns
            {
                "pattern": r"(?i)(presentation|demo|pitch).{0,30}(?:on|next|this|tomorrow)\s+([a-zA-Z\s]+)",
                "event_type": "presentation",
                "extract_fields": ["date"]
            },
            # Deadline patterns
            {
                "pattern": r"(?i)(deadline|due).{0,30}(?:on|next|this|by)\s+([a-zA-Z\s]+)",
                "event_type": "deadline",
                "extract_fields": ["date"]
            }
        ]
        
        # Extract emotional context
        emotional_indicators = {
            "nervous": ["nervous", "anxious", "worried", "stressed"],
            "excited": ["excited", "thrilled", "can't wait", "looking forward"],
            "concerned": ["concerned", "worried", "unsure", "doubt"],
            "confident": ["confident", "ready", "prepared", "sure"]
        }
        
        for pattern_info in event_patterns:
            matches = re.finditer(pattern_info["pattern"], conversation_text)
            
            for match in matches:
                try:
                    event = {
                        "event_type": pattern_info["event_type"],
                        "raw_text": match.group(0),
                        "participants": participants,
                        "extracted_at": datetime.utcnow().isoformat()
                    }
                    
                    # Extract specific fields based on pattern
                    groups = match.groups()
                    for i, field in enumerate(pattern_info["extract_fields"]):
                        if i + 1 < len(groups) and groups[i + 1]:
                            event[field] = groups[i + 1].strip()
                    
                    # Parse and normalize date if present
                    if "date" in event:
                        parsed_date = self._parse_relative_date(event["date"])
                        if parsed_date:
                            event["parsed_date"] = parsed_date.isoformat()
                            event["date_confidence"] = "high" if "next" in event["date"].lower() or "tomorrow" in event["date"].lower() else "medium"
                    
                    # Extract emotional context
                    for emotion, indicators in emotional_indicators.items():
                        if any(indicator in conversation_text.lower() for indicator in indicators):
                            event["initial_emotion"] = emotion
                            break
                    
                    # Generate reflection triggers
                    event["reflection_triggers"] = self._generate_reflection_triggers(event)
                    
                    events.append(event)
                    
                except Exception as e:
                    logger.warning(f"Error processing event match: {str(e)}")
                    continue
        
        return events
    
    def _parse_relative_date(self, date_str: str) -> Optional[datetime]:
        """Parse relative date expressions into actual dates"""
        try:
            date_str = date_str.lower().strip()
            now = datetime.now()
            
            # Handle common relative expressions
            if "tomorrow" in date_str:
                return now + timedelta(days=1)
            elif "next week" in date_str:
                return now + timedelta(weeks=1)
            elif "next month" in date_str:
                return now + relativedelta(months=1)
            elif "next friday" in date_str or "friday" in date_str:
                days_ahead = 4 - now.weekday()  # Friday is 4
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
            elif "next monday" in date_str or "monday" in date_str:
                days_ahead = 0 - now.weekday()  # Monday is 0
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
            elif "next tuesday" in date_str or "tuesday" in date_str:
                days_ahead = 1 - now.weekday()  # Tuesday is 1
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
            elif "next wednesday" in date_str or "wednesday" in date_str:
                days_ahead = 2 - now.weekday()  # Wednesday is 2
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
            elif "next thursday" in date_str or "thursday" in date_str:
                days_ahead = 3 - now.weekday()  # Thursday is 3
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
            elif "next saturday" in date_str or "saturday" in date_str:
                days_ahead = 5 - now.weekday()  # Saturday is 5
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
            elif "next sunday" in date_str or "sunday" in date_str:
                days_ahead = 6 - now.weekday()  # Sunday is 6
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
            else:
                # Try to parse with dateutil
                return parse_date(date_str, fuzzy=True)
                
        except Exception as e:
            logger.debug(f"Could not parse date '{date_str}': {str(e)}")
            return None
    
    def _generate_reflection_triggers(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate reflection triggers based on event type and timing"""
        triggers = []
        
        if "parsed_date" not in event:
            return triggers
            
        event_date = datetime.fromisoformat(event["parsed_date"])
        event_type = event.get("event_type", "")
        
        # Pre-event triggers
        if event_type in ["interview", "presentation", "meeting"]:
            # 3 days before for preparation
            triggers.append({
                "trigger_date": (event_date - timedelta(days=3)).isoformat(),
                "trigger_type": "pre_event",
                "message_template": "How are you feeling about your {event_type} {timing}?",
                "timing": "in 3 days"
            })
            
            # Day before for final preparation
            triggers.append({
                "trigger_date": (event_date - timedelta(days=1)).isoformat(),
                "trigger_type": "pre_event",
                "message_template": "Your {event_type} is tomorrow. How are you feeling about it?",
                "timing": "tomorrow"
            })
        
        # Post-event triggers
        triggers.append({
            "trigger_date": (event_date + timedelta(days=1)).isoformat(),
            "trigger_type": "post_event",
            "message_template": "How did your {event_type} go yesterday?",
            "timing": "yesterday"
        })
        
        # Follow-up triggers (for longer-term events)
        if event_type in ["interview"]:
            triggers.append({
                "trigger_date": (event_date + timedelta(weeks=1)).isoformat(),
                "trigger_type": "follow_up",
                "message_template": "Any updates on the {event_type} you had last week?",
                "timing": "last week"
            })
        
        return triggers
    
    async def _store_intelligent_memory(self, event: Dict[str, Any], participants: List[str]) -> Optional[Dict[str, Any]]:
        """Store the extracted event as an intelligent memory in MongoDB"""
        try:
            # Get primary user (first participant)
            primary_user_id = participants[0] if participants else "unknown"
            
            # Create memory document structure
            memory_data = {
                "type": "intelligent_event",
                "event_type": event.get("event_type", "unknown"),
                "content": self._create_memory_content(event),
                "created_by": primary_user_id,
                "participants": participants,
                "metadata": {
                    "extracted_data": event,
                    "reflection_triggers": event.get("reflection_triggers", []),
                    "initial_emotion": event.get("initial_emotion"),
                    "event_date": event.get("parsed_date"),
                    "extraction_confidence": event.get("date_confidence", "medium"),
                    "tags": [event.get("event_type", "event"), "intelligent_extraction"]
                },
                "visibility": {
                    "type": "family",
                    "shared_with": participants
                }
            }
            
            # Store in MongoDB
            memory_id = await self.mongo_manager.create_memory(memory_data)
            
            logger.info(f"Stored intelligent memory: {memory_id} for event: {event.get('event_type', 'unknown')}")
            
            return {
                "memory_id": memory_id,
                "event_type": event.get("event_type"),
                "event_date": event.get("parsed_date"),
                "reflection_triggers_count": len(event.get("reflection_triggers", []))
            }
            
        except Exception as e:
            logger.error(f"Error storing intelligent memory: {str(e)}")
            return None
    
    def _create_memory_content(self, event: Dict[str, Any]) -> str:
        """Create human-readable memory content from extracted event data"""
        event_type = event.get("event_type", "event")
        content_parts = [f"Extracted {event_type}:"]
        
        # Add key details based on event type
        if event_type == "interview":
            if "company" in event:
                content_parts.append(f"Company: {event['company']}")
            if "role" in event:
                content_parts.append(f"Role: {event['role']}")
        elif event_type == "trip":
            if "destination" in event:
                content_parts.append(f"Destination: {event['destination']}")
        elif event_type == "birthday":
            if "person" in event:
                content_parts.append(f"Person: {event['person']}")
        
        # Add date and emotion if available
        if "parsed_date" in event:
            date_obj = datetime.fromisoformat(event["parsed_date"])
            content_parts.append(f"Date: {date_obj.strftime('%Y-%m-%d')}")
        
        if "initial_emotion" in event:
            content_parts.append(f"Initial emotion: {event['initial_emotion']}")
        
        # Add original context
        if "raw_text" in event:
            content_parts.append(f"Context: \"{event['raw_text']}\"")
        
        return " | ".join(content_parts)

# Singleton instance
_memory_intelligence_service: Optional[MemoryIntelligenceService] = None

async def get_memory_intelligence_service() -> MemoryIntelligenceService:
    """Get singleton instance of MemoryIntelligenceService"""
    global _memory_intelligence_service
    if _memory_intelligence_service is None:
        _memory_intelligence_service = MemoryIntelligenceService()
        await _memory_intelligence_service.initialize()
    return _memory_intelligence_service 