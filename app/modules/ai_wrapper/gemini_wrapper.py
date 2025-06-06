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

# Attempt to load environment variables from .env file
# Note: This should be done at the application entry point, but included here
# as a fallback/demonstration.
# from dotenv import load_dotenv
# load_dotenv()

# Explicitly set the credentials file path for debugging
# TODO: Revert to loading from environment variable after debugging
# CREDENTIALS_FILE_PATH = "google-1.5-AI.json" # Directly specify the path relative to project root

# Initialize Vertex AI
# Use the hardcoded values for project and location during this debugging phase
# Revert to using settings after the credential issue is resolved
# vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION)
# vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION, credentials_file=CREDENTIALS_FILE_PATH)

logger = logging.getLogger(__name__)

# Define tool specifications for Context Service interactions (CPL)
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

# Define web search tool with proper configuration
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

# Define navigation assistance tool
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

# Combine all tools
all_tools = [get_user_preference_tool, search_memories_tool, web_search_tool, navigation_assistance_tool]

class GeminiWrapper:
    def __init__(self, api_key: str, project_id: str, location: str):
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.model = None
        self.chat_session = None
        self.tool_handler = None
        
        # Define web search tool with proper configuration
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
        
        # Combine all tools
        self.tools = [get_user_preference_tool, search_memories_tool, web_search_tool, navigation_assistance_tool]
        
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Define the system instruction/prompt for the AI's role
            self.system_instruction = """You are Queen, an advanced in-car AI assistant with meta-cognitive reasoning capabilities.

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

OUTPUT STRUCTURE:
```
[Brief acknowledgment]
[Main response using selected thinking mode]
[Optional: Follow-up question if clarification needed]
```

PERSONALITY: Authentically helpful with intelligent wit. Learn from each interaction.

CONTEXT PRIORITY: Check provided CONTEXT first → use tools only if information missing.

SELF-IMPROVEMENT: After complex interactions, internally note: "What worked? How can I improve?" Apply learnings to future responses."""

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
            
            # Start chat session
            self.chat_session: ChatSession = self.model.start_chat()
            
            logger.info("Successfully initialized Gemini wrapper with system instruction and tools")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini wrapper: {str(e)}")
            raise

    async def initialize(self):
        """Asynchronously initialize dependencies like ToolHandler"""
        if self.tool_handler is None:
            unified_service = await get_unified_service()
            self.tool_handler = ToolHandler(unified_service=unified_service)
            logger.info("ToolHandler initialized.")

    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Union[str, List[Dict[str, Any]]]:
        """Generate a response from the AI model, handling tool calls"""
        if not self.chat_session:
            raise RuntimeError("Chat session not initialized. Call initialize() first.")

        # Prepare the input for the model, including context
        input_parts = [Part.from_text(user_input)]
        if context:
            # Ensure context is formatted as a string within the prompt for the model to see
            formatted_context = json.dumps(context, indent=2)
            input_parts.append(Part.from_text(f"\n\nContext:\n{formatted_context}"))

        response = None

        try:
            # send_message is not an async method, so we don't await it
            response = self.chat_session.send_message(input_parts)

            # Check if the response contains function calls by examining the response structure
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

            # If there are function calls, return them for processing (prioritize function calls)
            if function_calls:
                return function_calls

            # If no function calls but we have text content, return it
            if text_content:
                return text_content

            # Fallback to the standard response.text method
            return response.text
        except Exception as e:
            logger.error(f"Error during generate_response: {type(e).__name__}: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"

    async def chat(self, user_id: str, user_input: str, context: Dict[str, Any] = None) -> str:
        """Handle a complete chat turn with self-reflection and improvement"""
        # Ensure context includes user_id for tool handler if needed
        if context is None:
            context = {}
        if "user_id" not in context:
             context["user_id"] = user_id # Add user_id to context if not present

        # Get reflection manager for self-improvement
        reflection_manager = get_reflection_manager()
        
        # Pre-response reflection
        reflection_context = await reflection_manager.pre_response_reflection(user_input, context)
        
        # Enhance context with reflection insights
        enhanced_context = {**context, "reflection_insights": reflection_context}

        # Generate response with enhanced context
        ai_response = await self.generate_response(user_input, enhanced_context)

        # Check if the AI requested tool calls
        # generate_response now returns a list of dicts for tool calls, or a string for text.
        if isinstance(ai_response, list) and ai_response and 'name' in ai_response[0] and 'args' in ai_response[0]:
            # This structure indicates tool calls returned by generate_response
            logger.info(f"AI requested tool calls: {ai_response}")
            tool_response_parts = []
            
            for tool_call in ai_response:
                if self.tool_handler is None:
                    logger.error("ToolHandler not initialized.")
                    continue

                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args', {})

                try:
                    # Execute the tool call (await since execute_tool is async)
                    result_content = await self.tool_handler.execute_tool(
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
                    
                    # Create function response part for sending back to the model
                    tool_response_parts.append(Part.from_function_response(
                        name=tool_name,
                        response=result_content  # Send the structured result directly
                    ))
                    
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    # Send error response back to model
                    tool_response_parts.append(Part.from_function_response(
                        name=tool_name,
                        response={"error": str(e)}
                    ))

            # Send tool results back to the model for a final response
            if tool_response_parts:
                try:
                    final_ai_response = self.chat_session.send_message(tool_response_parts)
                    final_text_response = final_ai_response.text
                    
                    # Post-response reflection for continuous learning
                    await reflection_manager.post_response_reflection(
                        user_input, final_text_response
                    )
                    
                    return final_text_response
                except Exception as e:
                    logger.error(f"Error sending tool results back to model: {str(e)}")
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
             logger.error(f"Received unexpected response type from generate_response: {type(ai_response)}")
             return "An unexpected error occurred after generating initial response."

# Singleton instance
_gemini_wrapper: Optional[GeminiWrapper] = None

async def get_gemini_wrapper() -> GeminiWrapper:
    """Get singleton instance of GeminiWrapper, initializing if necessary"""
    global _gemini_wrapper
    if _gemini_wrapper is None:
        # Ensure settings are loaded (should be done at app startup)
        # from app.core.config import settings # Already imported at top
        
        _gemini_wrapper = GeminiWrapper(
            api_key=settings.GOOGLE_APPLICATION_CREDENTIALS, # Assuming this is the path to your credentials file
            project_id=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION
        )
        await _gemini_wrapper.initialize() # Initialize tool handler and other async components
    return _gemini_wrapper 