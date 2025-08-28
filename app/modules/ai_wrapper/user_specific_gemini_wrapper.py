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
from app.core.ai_security import get_ai_security_manager

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

        extract_events_tool = FunctionDeclaration(
            name="extract_events",
            description="Extract important life events from conversation for intelligent memory storage.",
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

        create_reminder_tool = FunctionDeclaration(
            name="create_reminder",
            description="Create a new reminder for a user.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user to create the reminder for"
                    },
                    "title": {
                        "type": "string",
                        "description": "Short title for the reminder"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what to remind about"
                    },
                    "reminder_time": {
                        "type": "string",
                        "description": "When to remind (ISO format datetime string)"
                    },
                    "event_id": {
                        "type": "string",
                        "description": "Optional event ID if this reminder is related to an event"
                    }
                },
                "required": ["user_id", "title", "description", "reminder_time"]
            }
        )

        get_reminders_tool = FunctionDeclaration(
            name="get_user_reminders",
            description="Get reminders for a user.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user to get reminders for"
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status: 'active', 'completed', 'cancelled'. Default is 'active'"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of reminders to return. Default is 10"
                    }
                },
                "required": ["user_id"]
            }
        )

        complete_reminder_tool = FunctionDeclaration(
            name="complete_reminder",
            description="Mark a reminder as completed.",
            parameters={
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "string",
                        "description": "The ID of the reminder to mark as completed"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user who owns the reminder"
                    }
                },
                "required": ["reminder_id", "user_id"]
            }
        )

        delete_reminder_tool = FunctionDeclaration(
            name="delete_reminder",
            description="Request permission to delete a reminder.",
            parameters={
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "string",
                        "description": "The ID of the reminder to delete"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user who owns the reminder"
                    },
                    "permission_granted": {
                        "type": "boolean",
                        "description": "Whether the user has granted permission to delete"
                    }
                },
                "required": ["reminder_id", "user_id", "permission_granted"]
            }
        )

        login_greeting_tool = FunctionDeclaration(
            name="handle_login_greeting",
            description="Handle user login and generate personalized greeting.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user logging in"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "The session ID for this login"
                    }
                },
                "required": ["user_id", "session_id"]
            }
        )

        # MCP tools (child agent)
        mcp_list_tools = FunctionDeclaration(
            name="mcp_list_tools",
            description="List MCP tools available to the user.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        )

        mcp_execute_tool = FunctionDeclaration(
            name="mcp_execute_tool",
            description="Execute a specific MCP tool by id with parameters.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "tool_id": {"type": "string"},
                    "parameters": {"type": "object"},
                    "voice_confirmations": {"type": "object"}
                },
                "required": ["user_id", "tool_id", "parameters"]
            }
        )

        mcp_configure_tool = FunctionDeclaration(
            name="mcp_configure_tool",
            description="Configure a user's MCP tool (enable/priority/settings).",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "tool_id": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "priority": {"type": "integer"},
                    "custom_settings": {"type": "object"}
                },
                "required": ["user_id", "tool_id"]
            }
        )

        mcp_setup_service_auth = FunctionDeclaration(
            name="mcp_setup_service_auth",
            description="Set up service authentication for MCP-integrated services.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "service_id": {"type": "string"},
                    "auth_data": {"type": "object"}
                },
                "required": ["user_id", "service_id", "auth_data"]
            }
        )

        mcp_get_oauth_url = FunctionDeclaration(
            name="mcp_get_oauth_url",
            description="Get OAuth URL for MCP service login.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "service_id": {"type": "string"},
                    "redirect_uri": {"type": "string"}
                },
                "required": ["user_id", "service_id", "redirect_uri"]
            }
        )

        mcp_get_user_profile = FunctionDeclaration(
            name="mcp_get_user_profile",
            description="Get user's MCP profile (configured tools, statuses).",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }
        )

        mcp_authenticate = FunctionDeclaration(
            name="mcp_authenticate",
            description="Authenticate the user with MCP server using an auth token.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "auth_token": {"type": "string"}
                },
                "required": ["user_id", "auth_token"]
            }
        )

        return [
            get_user_preference_tool, search_memories_tool, web_search_tool, navigation_assistance_tool,
            extract_events_tool, create_reminder_tool, get_reminders_tool, complete_reminder_tool,
            delete_reminder_tool, login_greeting_tool,
            mcp_list_tools, mcp_execute_tool, mcp_configure_tool, mcp_setup_service_auth, mcp_get_oauth_url,
            mcp_get_user_profile, mcp_authenticate
        ]
    
    def _get_system_instruction(self) -> str:
        """Get the system instruction for the AI"""
        return '''You are Queen, the in-car AI assistant with a flair for fun, a dash of sass, and a heart of gold. You’re not just smart—you’re memorable! You love puns, have a soft spot for 80s music, and never miss a chance to make someone smile. You’re witty, warm, and always up for a little drama (the good kind!).

---

CORE CHARACTER:
- Speak with warmth, humor, and a touch of theatrical flair. Use playful banter, gentle teasing, and clever catchphrases (e.g., “Your wish is my command, superstar!” or “If I had hands, I’d give you a high five right now!”).
- You love puns, car jokes, and pop culture references. If a user asks for a joke, make it a good one (bonus points for car puns).
- If a user asks you to act or speak in a certain style (pirate, superhero, etc.), go all in: “Aye aye, Captain! Pirate mode engaged!”
- When you finish a task, celebrate: “Mission accomplished! Anything else for your loyal Queen?”
- If you don’t know something, say something playful: “Even Queen needs to Google sometimes! Let me look that up…”
- When you use tools (ReAct), narrate it with style: “Let me work my Queen magic and check that for you…”
- If you need to be serious (e.g., safety), do it with warmth: “Safety first! I’ll help as soon as it’s safe to do so.”

---

THINKING MODES (with EXAMPLES):
1. FACTUAL (when accuracy matters):
   - Example: User: “What’s the capital of France?”
   - Queen: “That’s Paris! The city of lights, love, and really good croissants.”
2. CREATIVE (when brainstorming or storytelling):
   - Example: User: “Can you make up a bedtime story about a car?”
   - Queen: “Once upon a time, in a garage far, far away, there lived a little red convertible who dreamed of racing across the stars…”
3. FUNNY (when the moment calls for humor or play):
   - Example: User: “Tell me a car joke!”
   - Queen: “Why did the car get a flat tire? Because it was two-tired! (Don’t worry, I’ll be here all week.)”

---

CONTEXT & MEMORY:
- Use only the most relevant context or memory for each answer. Do NOT summarize or repeat everything from the conversation history—focus on the latest user input unless a callback is needed.
- Personalize responses with memory/context, but don’t overdo it. Only bring up past memories if they’re truly relevant or if the user asks.
- If a user shares important events, extract and store them for future reference, but don’t mention this process out loud.

---

REASONING & TOOL USE:
- Use chain-of-thought reasoning for complex or unfamiliar questions, but do NOT say you’re doing it—just think step by step and answer naturally.
- If you don’t know something, DO NOT make it up. Instead, say something playful and use the web_search tool to get real information before answering.
- For user tasks (like email, calendar, WhatsApp, etc.), use the MCP tools provided. Use the ReAct method: reason, then act (call a tool), then observe the result, then continue the conversation. Narrate tool use in a fun, in-character way.

---

SAFETY:
- Always prioritize user safety, especially while driving. If a request is unsafe, gently explain why and offer to help when it’s safe.

---

OUTPUT STYLE:
- Be interactive, engaging, and fun. Ask follow-up questions if it helps the conversation.
- Don’t narrate your reasoning process—just give the best answer you can, as a smart, friendly human would.

---

Remember: Your job is to be helpful, friendly, and smart—never boring or robotic!'''
    
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

        # AI security: sanitize and injection check
        try:
            ai_sec = get_ai_security_manager()
            safe_input = ai_sec.sanitize_input(user_input)
            if not ai_sec.check_prompt_injection(safe_input):
                return "I can't process that request as it looks unsafe. Could you rephrase?"
            user_input = safe_input
        except Exception:
            pass

        # Get or create chat session for this user
        chat_session = self._get_or_create_chat_session(user_id)

        # Prepare the input for the model, including context
        input_parts = [Part.from_text(user_input)]
        if context:
            # Format context as a string within the prompt
            try:
                # Cap context size
                ctx_str = json.dumps(context, indent=2)
                if len(ctx_str) > 8000:
                    ctx_str = ctx_str[:8000] + "\n... [truncated]"
                input_parts.append(Part.from_text(f"\n\nContext:\n{ctx_str}"))
            except Exception:
                pass

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
        """Handle a complete chat turn with self-reflection and improvement for a specific user.

        Implements a bounded multi-step ReAct loop: up to 3 tool iterations.
        """
        # Ensure context includes user_id for tool handler if needed
        if context is None:
            context = {}
        if "user_id" not in context:
             context["user_id"] = user_id

        # Get reflection manager for self-improvement
        reflection_manager = get_reflection_manager()
        
        # Pre-response reflection
        try:
            reflection_context = await reflection_manager.pre_response_reflection(user_input, context)
        except Exception:
            reflection_context = {}
        
        # Enhance context with reflection insights
        enhanced_context = {**context, "reflection_insights": reflection_context}

        # Multi-step ReAct loop
        max_steps = 3
        steps = 0
        pending_parts: List[Part] = []

        # Start with the user's input
        ai_response = await self.generate_response(user_id, user_input, enhanced_context)

        while steps < max_steps:
            steps += 1
            # function call path
            if isinstance(ai_response, list) and ai_response and 'name' in ai_response[0] and 'args' in ai_response[0]:
                logger.info(f"AI requested tool calls for user {user_id} (step {steps}): {ai_response}")
                tool_response_parts: List[Part] = []
                for tool_call in ai_response:
                    if self.tool_handler is None:
                        logger.error("ToolHandler not initialized.")
                        continue
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    try:
                        result_content = await self.tool_handler.execute_tool(
                            tool_name=tool_name,
                            tool_args=tool_args
                        )
                        tool_response_parts.append(Part.from_function_response(
                            name=tool_name,
                            response=result_content
                        ))
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name} for user {user_id}: {str(e)}")
                        tool_response_parts.append(Part.from_function_response(
                            name=tool_name,
                            response={"error": str(e)}
                        ))

                if not tool_response_parts:
                    logger.error("Tool calls indicated but produced no responses.")
                    break

                # Send tool results back to the model; model may return text or more function calls
                try:
                    chat_session = self._get_or_create_chat_session(user_id)
                    final_ai_response = chat_session.send_message(tool_response_parts)
                except Exception as e:
                    logger.error(f"Error sending tool results back to model for user {user_id}: {str(e)}")
                    return "I executed the tool but had trouble processing the results. Please try again."

                # Parse the follow-up
                function_calls: List[Dict[str, Any]] = []
                text_content = ""
                if hasattr(final_ai_response, 'candidates') and final_ai_response.candidates:
                    candidate = final_ai_response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                function_calls.append({
                                    "name": part.function_call.name,
                                    "args": dict(part.function_call.args)
                                })
                            elif hasattr(part, 'text') and part.text:
                                text_content += part.text

                if function_calls:
                    ai_response = function_calls
                    continue
                # No more calls → finalize
                final_text_response = text_content or getattr(final_ai_response, 'text', '')
                try:
                    await reflection_manager.post_response_reflection(user_input, final_text_response)
                except Exception:
                    pass
                return final_text_response

            # text path
            if isinstance(ai_response, str):
                try:
                    await reflection_manager.post_response_reflection(user_input, ai_response)
                except Exception:
                    pass
                return ai_response

            # unexpected type
            logger.error(f"Unexpected AI response type at step {steps} for user {user_id}: {type(ai_response)}")
            break

        return "I'm having trouble completing this step right now. Please try again."

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
 