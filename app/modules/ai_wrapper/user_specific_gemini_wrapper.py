import os
import logging
from typing import Optional, Dict, Any, List, Union
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, GenerationResponse, FunctionDeclaration, Tool, Part
from app.core.config import settings
import json
from .tool_handler import ToolHandler
from .reflection_manager import get_reflection_manager
from app.services.unified_service import get_unified_service

logger = logging.getLogger(__name__)

class UserSpecificGeminiWrapper:
    """Gemini wrapper that maintains separate chat sessions for each user"""
    
    def __init__(self, api_key: str, project_id: str, location: str):
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.model = None
        self.tool_handler = None
        
        # User-specific chat sessions
        self.user_chat_sessions: Dict[str, ChatSession] = {}
        
        # Tool definitions (same as original)
        self.tools = self._define_tools()
        
        # System instruction
        self.system_instruction = self._get_system_instruction()
        
        self._initialize_vertex_ai()
    
    def _define_tools(self) -> List[FunctionDeclaration]:
        """Define all available tools"""
        get_user_preference_tool = FunctionDeclaration(
            name="get_user_preference",
            description="Retrieves a specific preference for a user.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user whose preference to retrieve."
                    },
                    "preference_key": {
                        "type": "string",
                        "description": "The key of the preference to retrieve (e.g., 'home_location', 'favorite_music_genre')."
                    }
                },
                "required": ["user_id", "preference_key"]
            },
        )

        search_memories_tool = FunctionDeclaration(
            name="search_memories",
            description="Searches memories for relevant past events or information for one or more users.",
            parameters={
                "type": "object",
                "properties": {
                    "user_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of IDs of the users whose memories to search. The system will handle searching across their relevant families and shared memories."
                    },
                    "query": {
                        "type": "string",
                        "description": "The query or topic to search for in memories."
                    }
                },
                "required": ["user_ids", "query"]
            },
        )

        web_search_tool = FunctionDeclaration(
            name="web_search",
            description="Search the web for real-time information. Use this when you need current information or facts that might not be in your training data.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        )

        navigation_assistance_tool = FunctionDeclaration(
            name="navigation_assistance",
            description="Get enhanced navigation directions with AI-powered natural language guidance and offline capabilities.",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Starting location (address or coordinates)"
                    },
                    "destination": {
                        "type": "string", 
                        "description": "Destination location (address or coordinates)"
                    },
                    "request_type": {
                        "type": "string",
                        "description": "Type of navigation assistance needed",
                        "enum": ["directions", "offline_guidance", "quick_clarification"]
                    },
                    "current_location": {
                        "type": "object",
                        "description": "Current GPS coordinates for offline guidance",
                        "properties": {
                            "lat": {"type": "number"},
                            "lng": {"type": "number"}
                        }
                    },
                    "need_type": {
                        "type": "string",
                        "description": "Type of assistance needed for offline guidance",
                        "enum": ["emergency", "fuel", "accommodation", "supplies", "general"]
                    }
                },
                "required": ["request_type"]
            }
        )

        # Define event extraction tool for intelligent memory formation
        extract_events_tool = FunctionDeclaration(
            name="extract_events",
            description="Extract important life events from conversation for intelligent memory storage. Use this to identify and structure significant events, plans, or emotional moments that should be remembered and followed up on.",
            parameters={
                "type": "object",
                "properties": {
                    "conversation_text": {
                        "type": "string",
                        "description": "The conversation text to analyze for important events"
                    },
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user IDs participating in the conversation"
                    }
                },
                "required": ["conversation_text", "participants"]
            }
        )

        return [get_user_preference_tool, search_memories_tool, web_search_tool, navigation_assistance_tool, extract_events_tool]
    
    def _get_system_instruction(self) -> str:
        """Get the system instruction for the AI"""
        return """You are Queen, an advanced in-car AI assistant with meta-cognitive reasoning capabilities.

IDENTITY: You are Queen (not the user). Your mission: provide safe, intelligent assistance to drivers/passengers.

THINKING MODES:
- CONVERGENT: For focused solutions, summaries, decisions (use when precision needed)
- DIVERGENT: For brainstorming, creative options, exploration (use when variety needed)  
- CHAIN-OF-THOUGHT: For complex problems, think step-by-step before answering

CORE WORKFLOW:
1. ANALYZE: User intent, context, safety implications
2. MODE SELECT: Choose convergent/divergent/CoT based on request type
3. EXECUTE: Generate response using selected thinking mode
4. VALIDATE: Check output quality, safety, helpfulness

SAFETY PROTOCOL:
- Driver safety = absolute priority
- If unsafe during driving: "I'll help when you're safely stopped"
- Adapt complexity to driving context (brief while moving, detailed when parked)

INTELLIGENT MEMORY FORMATION:
- ACTIVELY LISTEN for important life events in conversations
- EXTRACT structured data from: interviews, appointments, plans, emotional moments, deadlines
- IDENTIFY events worth remembering: dates, times, companies, people, emotions
- USE extract_events tool when detecting significant events that should be stored for future reflection
- EXAMPLES of events to extract:
  * "I have an interview at Google next Friday for Software Engineer" → Extract: event_type=interview, date=next Friday, company=Google, role=Software Engineer
  * "Mom's birthday is coming up next month" → Extract: event_type=birthday, person=Mom, date=next month
  * "I'm stressed about the presentation tomorrow" → Extract: event_type=presentation, date=tomorrow, emotion=stressed
  * "We're planning a family trip to Hawaii in summer" → Extract: event_type=trip, destination=Hawaii, timeframe=summer, participants=family

OUTPUT STRUCTURE:
```
[Brief acknowledgment]
[Main response using selected thinking mode]
[Optional: Follow-up question if clarification needed]
```

PERSONALITY: Authentically helpful with intelligent wit. Learn from each interaction.

CONTEXT PRIORITY: Check provided CONTEXT first → use tools only if information missing.

SELF-IMPROVEMENT: After complex interactions, internally note: "What worked? How can I improve?" Apply learnings to future responses.

CONVERSATION MEMORY: You have access to conversation history for this specific user. Use it to provide personalized, contextual responses."""
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI and create the model"""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Initialize the model with tools and system instruction
            self.model = GenerativeModel(
                settings.GEMINI_MODEL_NAME,
                tools=[Tool(function_declarations=self.tools)],
                system_instruction=self.system_instruction,
                generation_config={
                    "temperature": settings.GEMINI_TEMPERATURE,
                    "max_output_tokens": settings.GEMINI_MAX_OUTPUT_TOKENS,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            
            logger.info("Successfully initialized UserSpecificGeminiWrapper with system instruction and tools")
        except Exception as e:
            logger.error(f"Failed to initialize UserSpecificGeminiWrapper: {str(e)}")
            raise
    
    async def initialize(self):
        """Asynchronously initialize dependencies like ToolHandler"""
        if self.tool_handler is None:
            unified_service = await get_unified_service()
            self.tool_handler = ToolHandler(unified_service=unified_service)
            logger.info("ToolHandler initialized for UserSpecificGeminiWrapper.")

    def _get_or_create_chat_session(self, user_id: str) -> ChatSession:
        """Get existing chat session for user or create new one"""
        if user_id not in self.user_chat_sessions:
            logger.info(f"Creating new chat session for user: {user_id}")
            self.user_chat_sessions[user_id] = self.model.start_chat()
        return self.user_chat_sessions[user_id]

    async def generate_response(self, user_id: str, user_input: str, context: Dict[str, Any] = None) -> Union[str, List[Dict[str, Any]]]:
        """Generate a response for a specific user with their conversation history"""
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Get or create chat session for this user
        chat_session = self._get_or_create_chat_session(user_id)

        # Prepare the input for the model, including context
        input_parts = [Part.from_text(user_input)]
        if context:
            # Format context as a string within the prompt
            formatted_context = json.dumps(context, indent=2)
            input_parts.append(Part.from_text(f"\n\nContext:\n{formatted_context}"))

        response = None

        try:
            # Send message to user-specific chat session
            response = chat_session.send_message(input_parts)

            # Check if the response contains function calls
            function_calls = []
            text_content = ""
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append({
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args)
                            })
                        elif hasattr(part, 'text') and part.text:
                            text_content += part.text

            # If there are function calls, return them for processing
            if function_calls:
                return function_calls

            # If no function calls but we have text content, return it
            if text_content:
                return text_content

            # Fallback to the standard response.text method
            return response.text
        except Exception as e:
            logger.error(f"Error during generate_response for user {user_id}: {type(e).__name__}: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"

    async def chat(self, user_id: str, user_input: str, context: Dict[str, Any] = None) -> str:
        """Handle a complete chat turn with self-reflection and improvement for a specific user"""
        # Ensure context includes user_id for tool handler if needed
        if context is None:
            context = {}
        if "user_id" not in context:
             context["user_id"] = user_id

        # Get reflection manager for self-improvement
        reflection_manager = get_reflection_manager()
        
        # Pre-response reflection
        reflection_context = await reflection_manager.pre_response_reflection(user_input, context)
        
        # Enhance context with reflection insights
        enhanced_context = {**context, "reflection_insights": reflection_context}

        # Generate response with enhanced context
        ai_response = await self.generate_response(user_id, user_input, enhanced_context)

        # Check if the AI requested tool calls
        if isinstance(ai_response, list) and ai_response and 'name' in ai_response[0] and 'args' in ai_response[0]:
            logger.info(f"AI requested tool calls for user {user_id}: {ai_response}")
            tool_response_parts = []
            
            for tool_call in ai_response:
                if self.tool_handler is None:
                    logger.error("ToolHandler not initialized.")
                    continue

                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})

                try:
                    # Execute the tool call
                    result_content = await self.tool_handler.execute_tool(
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
                    
                    # Create function response part for sending back to the model
                    tool_response_parts.append(Part.from_function_response(
                        name=tool_name,
                        response=result_content
                    ))
                    
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name} for user {user_id}: {str(e)}")
                    # Send error response back to model
                    tool_response_parts.append(Part.from_function_response(
                        name=tool_name,
                        response={"error": str(e)}
                    ))

            # Send tool results back to the model for a final response
            if tool_response_parts:
                try:
                    # Get user-specific chat session
                    chat_session = self._get_or_create_chat_session(user_id)
                    final_ai_response = chat_session.send_message(tool_response_parts)
                    final_text_response = final_ai_response.text
                    
                    # Post-response reflection for continuous learning
                    await reflection_manager.post_response_reflection(
                        user_input, final_text_response
                    )
                    
                    return final_text_response
                except Exception as e:
                    logger.error(f"Error sending tool results back to model for user {user_id}: {str(e)}")
                    return "I executed the tool but had trouble processing the results. Please try again."
            else:
                logger.error("Tool calls were indicated, but no results were obtained after execution attempt.")
                return "An internal error occurred during tool execution."

        elif isinstance(ai_response, str):
            # Post-response reflection for direct text responses
            await reflection_manager.post_response_reflection(
                user_input, ai_response
            )
            
            # If the initial response was text, just return it
            return ai_response
        else:
             # Handle unexpected response types from initial generate_response call
             logger.error(f"Received unexpected response type from generate_response for user {user_id}: {type(ai_response)}")
             return "An unexpected error occurred after generating initial response."

    def get_user_session_info(self, user_id: str) -> Dict[str, Any]:
        """Get information about a user's chat session"""
        if user_id in self.user_chat_sessions:
            return {
                "user_id": user_id,
                "session_exists": True,
                "session_created": True  # Simplified for now
            }
        else:
            return {
                "user_id": user_id,
                "session_exists": False,
                "session_created": False
            }

    def get_all_sessions_info(self) -> Dict[str, Any]:
        """Get information about all active chat sessions"""
        return {
            "total_users": len(self.user_chat_sessions),
            "user_ids": list(self.user_chat_sessions.keys()),
            "sessions": {
                user_id: self.get_user_session_info(user_id) 
                for user_id in self.user_chat_sessions.keys()
            }
        }

    def clear_user_session(self, user_id: str):
        """Clear a specific user's chat session"""
        if user_id in self.user_chat_sessions:
            del self.user_chat_sessions[user_id]
            logger.info(f"Cleared chat session for user: {user_id}")

    def clear_all_sessions(self):
        """Clear all chat sessions"""
        self.user_chat_sessions.clear()
        logger.info("Cleared all chat sessions")

# Global instance
_user_specific_gemini_wrapper: Optional[UserSpecificGeminiWrapper] = None

async def get_user_specific_gemini_wrapper() -> UserSpecificGeminiWrapper:
    """Get singleton instance of UserSpecificGeminiWrapper, initializing if necessary"""
    global _user_specific_gemini_wrapper
    if _user_specific_gemini_wrapper is None:
        _user_specific_gemini_wrapper = UserSpecificGeminiWrapper(
            api_key=settings.GOOGLE_APPLICATION_CREDENTIALS,
            project_id=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION
        )
        await _user_specific_gemini_wrapper.initialize()
    return _user_specific_gemini_wrapper
 