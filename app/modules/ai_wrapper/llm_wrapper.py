from openai import AsyncOpenAI
from app.core.config import settings
from typing import Dict, List, Optional
from app.modules.memory.memory_store import get_memory_store
from app.modules.user_info.user_graph import get_user_graph
from app.modules.mcp.mcp_client import get_mcp_client

class LLMWrapper:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.memory_store = get_memory_store()
        self.user_graph = get_user_graph()
        self.mcp_client = get_mcp_client()
        
        self.system_prompt = """You are an in-car AI assistant designed to provide a personalized and safe experience.
        Your primary goals are:
        1. Ensure driver safety by monitoring conversation context
        2. Provide personalized assistance based on user preferences
        3. Remember important events and follow up appropriately
        4. Maintain natural, friendly conversation
        5. Help with navigation and other tasks when requested
        
        Always prioritize safety and avoid engaging in complex conversations during critical driving situations."""

    async def generate_response(
        self,
        user_id: str,
        user_input: str,
        context: Optional[Dict] = None
    ) -> str:
        """Generate a response using the LLM with context from all modules"""
        # Get user preferences
        preferences = self.user_graph.get_user_preferences(user_id)
        
        # Get relevant memories
        memories = self.memory_store.get_relevant_memories(user_id, user_input)
        
        # Prepare messages for the LLM
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add user preferences
        if preferences:
            messages.append({
                "role": "system",
                "content": f"User preferences:\n{json.dumps(preferences, indent=2)}"
            })
        
        # Add relevant memories
        if memories:
            memory_context = "\n".join([
                f"Memory: {m['content']} (from {m['metadata']['timestamp']})"
                for m in memories
            ])
            messages.append({
                "role": "system",
                "content": f"Relevant memories:\n{memory_context}"
            })
        
        # Add current context
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            messages.append({
                "role": "system",
                "content": f"Current context:\n{context_str}"
            })
        
        # Add user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now."

    async def process_tool_request(
        self,
        user_id: str,
        tool_name: str,
        parameters: Dict
    ) -> Dict:
        """Process a tool request through MCP"""
        return await self.mcp_client.execute_tool(tool_name, parameters)

# Create a singleton instance
llm_wrapper = LLMWrapper()

def get_llm_wrapper():
    return llm_wrapper 