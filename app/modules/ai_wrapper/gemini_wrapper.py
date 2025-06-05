import os
import logging
from typing import Optional, Dict, Any, List, Union
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, GenerationResponse, FunctionDeclaration, Tool, Part
from app.core.config import settings
import json

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
vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION)
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

# Combine with existing tools like web search
all_tools = [get_user_preference_tool, search_memories_tool]

class GeminiWrapper:
    def __init__(self):
        """Initialize Gemini wrapper with Vertex AI and tools"""
        try:
            # Define the system instruction/prompt for the AI's role
            self.system_instruction = """You are an in-car AI assistant named 'Queen'.\n"\
                                      "Your primary goal is to assist the driver and passengers safely and effectively.\n"\
                                      "Focus on providing relevant information, controlling car features, and engaging in helpful conversation.\n"\
                                      "Prioritize driver safety at all times. Avoid distracting the driver during critical maneuvers.\n"\
                                      "Be polite, friendly, and concise. Use information about the user and their preferences when available to personalize responses.\n"\
                                      "Incorporate a slightly funny and quirky tone in your responses, but remain helpful and professional.\n" \
                                      "Pay attention to the user's mood. If the user seems low, sad, or distressed, respond with empathy and try to be comforting like a supportive friend, while still maintaining your assistant role.\n" \
                                      "You can access user preferences, recent interactions, and potentially other context provided to you.\n"\
                                      "If a request seems unsafe or distracting, politely decline and remind the user about safety.\n"\
                                      "---\n"\
                                      "---Context & Tool Usage---\n"\
                                      "Before deciding to use a tool, FIRST check the provided CONTEXT section for the information you need. If the information is present in the CONTEXT, use it directly.\n"\
                                      "If the information is NOT in the CONTEXT and the user is asking for their preferences or memories, YOU MUST use the appropriate tool (get_user_preference or search_memories) to retrieve the information. Do not answer based on general knowledge or admit you don't have the information without attempting to use the tool first.\n"\
                                      "You have access to tools to help you, especially for finding up-to-date information and recalling user-specific details.\n"\
                                      "When the user asks for information you don't have readily available, use the appropriate tool (e.g., get_user_preference, search_memories, web_search) to retrieve it.\n"\
                                      "Once you have the information from the tool, use it to formulate your response.\n"\
                                      """
            
            # Define available tools (using the combined list)
            self.tools = [Tool(function_declarations=all_tools)]

            # Initialize the model with tools and system instruction
            self.model = GenerativeModel(settings.GEMINI_MODEL_NAME, tools=self.tools, system_instruction=self.system_instruction)
            
            # Start chat
            self.chat: ChatSession = self.model.start_chat(
                # System instruction and tools are now passed during model initialization
                # tools=self.tools # Ensure tools are included on start_chat as well
            )
            
            logger.info("Successfully initialized Gemini wrapper with system instruction and tools")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini wrapper: {str(e)}")
            raise

    async def generate_response(
        self,
        user_id: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None # Add parameter for tool results
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Generate a response using Gemini model, potentially incorporating tool results.
        
        Args:
            user_id: The ID of the user.
            user_input: The user's input text (or the original input if responding to tool results).
            context: Optional context dictionary.
            tool_results: Optional list of results from previous tool calls.
        
        Returns:
            Union[str, List[Dict[str, Any]]]: The generated text response or a list of tool calls.
        """
        try:
            # Prepare the content to send to the chat session
            content_parts: List[Union[str, Part]] = []
            
            if tool_results:
                # If responding to tool results, send them as FunctionResponse Parts
                logger.info(f"Sending tool results to Gemini: {tool_results}")
                for result in tool_results:
                    content_parts.append(Part.from_function_response(
                        name=result["name"],
                        response={
                            "content": result["content"]
                            # Include other relevant fields from the tool result if needed
                        }
                    ))
                # If tool results are provided, the user_input here is likely the original query
                # that led to the tool call, so we might need to add it back for context.
                # Or, the calling logic should manage the history.
                # For now, let's assume the calling logic ensures the chat history is maintained
                # and just send the tool results. If a new turn, the user_input will be handled below.
            
            # Add user input (only if not primarily sending tool results, or if it's a new turn)
            # This logic might need refinement based on how chat history is managed.
            # For this implementation, we assume user_input is the primary input for a new turn.
            if user_input and not tool_results:
                 content_parts.append(user_input)

            # Add additional context if provided (can be sent alongside user input or tool results)
            if context:
                context_str = "Context:\n"
                for key, value in context.items():
                    context_str += f"{key}: {json.dumps(value, indent=2)}\n"
                content_parts.append(context_str)

            if not content_parts:
                 logger.warning("No content parts to send to Gemini.")
                 return "An internal error occurred: No content to send to AI."

            # Send content to the chat session
            response: GenerationResponse = await self.chat.send_message_async(content_parts)
            
            # Process and return the response - check for tool calls or text
            if response.candidates and response.candidates[0].content.parts:
                tool_calls = []
                text_response_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Ensure function call args are serializable
                        args_dict = {}
                        if hasattr(part.function_call, 'args') and part.function_call.args:
                            for arg_name, arg_value in part.function_call.args.items():
                                try:
                                    args_dict[arg_name] = json.loads(json.dumps(arg_value)) # Ensure JSON serializable
                                except (TypeError, json.JSONDecodeError):
                                    logger.warning(f"Argument {arg_name} is not JSON serializable: {arg_value}")
                                    args_dict[arg_name] = str(arg_value) # Fallback to string

                        tool_calls.append({
                            "name": part.function_call.name,
                            "args": args_dict
                        })
                    if hasattr(part, 'text') and part.text:
                        text_response_parts.append(part.text)
                
                if tool_calls:
                    # Return tool calls if any are present
                    logger.info(f"Gemini requested tool calls: {tool_calls}")
                    # Note: Returning the list of tool calls for the caller to execute
                    return tool_calls
                elif text_response_parts:
                    # Join text parts if only text is present
                    text = " ".join(text_response_parts)
                    logger.info(f"Received Gemini text response (first 50 chars): {text[:50]}...")
                    return text.strip()
                else:
                     logger.warning("Gemini response contains parts but no text or function_call.")
                     return "I received a response but couldn't understand it."
            else:
                # Handle cases where there are no candidates or parts (e.g., safety blocks)
                logger.warning(f"Gemini response has no candidates or parts: {response}")
                feedback = getattr(response, 'prompt_feedback', None)
                if feedback and feedback.block_reason:
                     logger.warning(f"Response blocked: {feedback.block_reason}")
                     return "I'm sorry, I can't respond to that request due to safety concerns."
                elif feedback and feedback.safety_ratings:
                     logger.warning(f"Response flagged by safety ratings: {feedback.safety_ratings}")
                     return "I'm sorry, I can't respond to that request."
                else:
                    return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_detail = str(e)
            return f"I apologize, but I'm having trouble processing your request right now. Details: {error_detail}"

    def _prepare_prompt(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare the prompt with context (deprecated for primary use).
        Note: This is now primarily for formatting context if needed outside generate_response.
        """
        logger.warning("_prepare_prompt method is primarily for formatting context in generate_response now.")
        formatted_context = ""
        if context:
             formatted_context += "Context:\n"
             for key, value in context.items():
                 formatted_context += f"{key}: {json.dumps(value, indent=2)}\n"
        
        return f"User Input: {user_input}\n\n{formatted_context.strip()}"

    def _process_response(self, response: GenerationResponse) -> str:
        """
        Process and clean the model's text response.
        Note: This method is now only for processing text responses after checking for tool calls.
        """
        try:
            text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')]
            text = " ".join(text_parts)

            # Basic cleaning
            text = text.strip()
            
            return text
        except Exception as e:
            logger.error(f"Error processing text response parts: {str(e)}")
            response_text_fallback = getattr(response, 'text', str(response))
            return f"I apologize, but I'm having trouble processing the response text. Details: {response_text_fallback}"

    async def reset_chat(self):
        """Reset the chat session"""
        try:
            # Restart chat, system instruction and tools are on the model
            self.chat = self.model.start_chat()
            logger.info("Chat session reset successfully with system instruction and tools")
        except Exception as e:
            logger.error(f"Error resetting chat session: {str(e)}")
            raise

# Singleton instance
_gemini_wrapper = None

async def get_gemini_wrapper() -> GeminiWrapper:
    """Get singleton instance of GeminiWrapper"""
    global _gemini_wrapper
    if _gemini_wrapper is None:
        _gemini_wrapper = GeminiWrapper()
    return _gemini_wrapper 