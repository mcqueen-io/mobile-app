from typing import Dict, Any, List
from app.services.unified_service import UnifiedService
from app.services.memory_intelligence_service import get_memory_intelligence_service
import aiohttp
import json
from app.core.config import settings

class ToolHandler:
    def __init__(self, unified_service: UnifiedService):
        self.unified_service = unified_service
        self.search_api_key = settings.GOOGLE_SEARCH_API_KEY  # You'll need to add this to your settings

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool call and return the result"""
        if tool_name == "get_user_preference":
            return await self.unified_service.get_user_preference(
                user_id=tool_args["user_id"],
                category=tool_args["category"]
            )
        elif tool_name == "search_memories":
            return await self.unified_service.search_memories(
                user_id=tool_args["user_id"],
                query=tool_args["query"]
            )
        elif tool_name == "web_search":
            return await self._execute_web_search(
                query=tool_args["query"],
                num_results=tool_args.get("num_results", 5)
            )
        elif tool_name == "navigation_assistance":
            return await self._execute_navigation_assistance(tool_args)
        elif tool_name == "extract_events":
            return await self._execute_event_extraction(tool_args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _execute_web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute a web search using Google Custom Search API and return structured data"""
        try:
            # Check if we have the required API credentials
            if not self.search_api_key or not hasattr(settings, 'GOOGLE_SEARCH_ENGINE_ID'):
                return {
                    "success": False,
                    "error": "Google Search API credentials not configured",
                    "query": query,
                    "results": []
                }

            async with aiohttp.ClientSession() as session:
                # Using Google Custom Search API
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": self.search_api_key,
                    "cx": settings.GOOGLE_SEARCH_ENGINE_ID,
                    "q": query,
                    "num": min(num_results, 10)  # Google API max is 10
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("items", [])
                        
                        # Structure the results for better AI processing
                        formatted_results = []
                        for item in results:
                            formatted_results.append({
                                "title": item.get("title", ""),
                                "link": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                                "displayLink": item.get("displayLink", "")
                            })
                        
                        # Return structured data that AI can process effectively
                        return {
                            "success": True,
                            "query": query,
                            "total_results": data.get("searchInformation", {}).get("totalResults", "0"),
                            "results": formatted_results,
                            "summary": self._create_search_summary(query, formatted_results)
                        }
                    else:
                        error_data = await response.json() if response.content_type == 'application/json' else await response.text()
                        return {
                            "success": False,
                            "error": f"API error (status {response.status}): {error_data}",
                            "query": query,
                            "results": []
                        }
                        
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "query": query,
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "query": query,
                "results": []
            }

    def _create_search_summary(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Create a comprehensive summary from search results for AI processing"""
        if not results:
            return f"No search results found for '{query}'."
        
        summary_parts = [
            f"Web search results for '{query}':",
            f"Found {len(results)} relevant results:",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description available")
            source = result.get("displayLink", "Unknown source")
            
            summary_parts.append(f"{i}. **{title}** (Source: {source})")
            summary_parts.append(f"   {snippet}")
            summary_parts.append("")
        
        # Add instruction for AI on how to use this data
        summary_parts.append("---")
        summary_parts.append("Use the above information to provide a comprehensive answer to the user's query.")
        
        return "\n".join(summary_parts)

    async def _execute_navigation_assistance(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute navigation assistance requests"""
        try:
            request_type = tool_args.get("request_type")
            
            if request_type == "directions":
                # Handle enhanced directions request
                from app.services.smart_navigation_service import get_smart_navigation_service
                nav_service = await get_smart_navigation_service()
                
                enhanced_directions = await nav_service.enhance_directions(
                    origin=tool_args.get("origin", ""),
                    destination=tool_args.get("destination", "")
                )
                
                if "error" in enhanced_directions:
                    return {"success": False, "error": enhanced_directions["error"]}
                
                # Format for AI response
                return {
                    "success": True,
                    "request_type": "directions",
                    "route_summary": f"Route from {tool_args.get('origin')} to {tool_args.get('destination')}",
                    "distance": enhanced_directions.get("total_distance", "Unknown"),
                    "duration": enhanced_directions.get("total_duration", "Unknown"),
                    "enhanced_steps": enhanced_directions.get("enhanced_steps", []),
                    "proactive_guidance": enhanced_directions.get("proactive_guidance", {}),
                    "confusing_areas": enhanced_directions.get("confusing_areas_count", 0)
                }
                
            elif request_type == "offline_guidance":
                # Handle offline guidance request
                from app.services.offline_navigation_cache import get_offline_navigation_cache
                offline_cache = await get_offline_navigation_cache()
                
                current_location = tool_args.get("current_location", {})
                need_type = tool_args.get("need_type", "general")
                
                guidance = await offline_cache.get_offline_guidance(
                    current_location=current_location,
                    user_context={"need_type": need_type}
                )
                
                return {
                    "success": True,
                    "request_type": "offline_guidance",
                    "offline_mode": True,
                    "guidance_message": guidance.get("guidance_message", ""),
                    "nearby_locations": guidance.get("nearby_essential_locations", []),
                    "simplified_directions": guidance.get("simplified_directions", [])
                }
                
            elif request_type == "quick_clarification":
                # Handle quick clarification
                from app.services.smart_navigation_service import get_smart_navigation_service
                nav_service = await get_smart_navigation_service()
                
                # This would need the instruction to clarify - simplified for now
                return {
                    "success": True,
                    "request_type": "quick_clarification", 
                    "message": "Please provide the specific navigation instruction you need help with."
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Unknown navigation request type: {request_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Navigation assistance error: {str(e)}"
            }

    async def _execute_event_extraction(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intelligent event extraction from conversation text"""
        try:
            conversation_text = tool_args.get("conversation_text", "")
            participants = tool_args.get("participants", [])
            
            if not conversation_text.strip():
                return {
                    "success": False,
                    "error": "No conversation text provided for event extraction"
                }
            
            if not participants:
                return {
                    "success": False,
                    "error": "No participants provided for event extraction"
                }
            
            # Get the Memory Intelligence Service
            memory_service = await get_memory_intelligence_service()
            
            # Extract and store events
            result = await memory_service.extract_and_store_events(
                conversation_text=conversation_text,
                participants=participants
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Event extraction error: {str(e)}",
                "events_found": 0
            }