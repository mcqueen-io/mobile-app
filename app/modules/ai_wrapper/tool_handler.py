from typing import Dict, Any, List
from app.services.unified_service import UnifiedService
from app.services.memory_intelligence_service import get_memory_intelligence_service
from app.services.reminder_service import get_reminder_service
from app.services.login_greeting_service import get_login_greeting_service
import aiohttp
import json
from app.core.config import settings
from datetime import datetime
from app.modules.mcp.mcp_agent import MCPChildAgent

class ToolHandler:
    def __init__(self, unified_service: UnifiedService):
        self.unified_service = unified_service
        self.search_api_key = settings.GOOGLE_SEARCH_API_KEY  # You'll need to add this to your settings
        self.mcp_agent = MCPChildAgent()

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
            # Ensure num_results is always an int
            num_results = tool_args.get("num_results", 5)
            try:
                num_results = int(num_results)
            except (ValueError, TypeError):
                num_results = 5
            return await self._execute_web_search(
                query=tool_args["query"],
                num_results=num_results
            )
        elif tool_name == "navigation_assistance":
            # Ensure any numeric arguments are cast to the correct type
            tool_args = tool_args.copy()
            if "confusing_areas_count" in tool_args:
                try:
                    tool_args["confusing_areas_count"] = int(tool_args["confusing_areas_count"])
                except (ValueError, TypeError):
                    tool_args["confusing_areas_count"] = 0
            return await self._execute_navigation_assistance(tool_args)
        elif tool_name == "extract_events":
            return await self._execute_event_extraction(tool_args)
        elif tool_name == "create_reminder":
            return await self._execute_create_reminder(tool_args)
        elif tool_name == "get_user_reminders":
            return await self._execute_get_reminders(tool_args)
        elif tool_name == "complete_reminder":
            return await self._execute_complete_reminder(tool_args)
        elif tool_name == "delete_reminder":
            return await self._execute_delete_reminder(tool_args)
        elif tool_name == "handle_login_greeting":
            return await self._execute_login_greeting(tool_args)
        elif tool_name == "mcp_list_tools":
            return await self._execute_mcp_list_tools(tool_args)
        elif tool_name == "mcp_execute_tool":
            return await self._execute_mcp_execute_tool(tool_args)
        elif tool_name == "mcp_configure_tool":
            return await self._execute_mcp_configure_tool(tool_args)
        elif tool_name == "mcp_setup_service_auth":
            return await self._execute_mcp_setup_service_auth(tool_args)
        elif tool_name == "mcp_get_oauth_url":
            return await self._execute_mcp_get_oauth_url(tool_args)
        elif tool_name == "mcp_get_user_profile":
            return await self._execute_mcp_get_user_profile(tool_args)
        elif tool_name == "mcp_authenticate":
            return await self._execute_mcp_authenticate(tool_args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _execute_mcp_list_tools(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        user_id = tool_args.get("user_id")
        return await self.mcp_agent.list_tools(user_id)

    async def _execute_mcp_execute_tool(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        user_id = tool_args.get("user_id")
        tool_id = tool_args.get("tool_id")
        parameters = tool_args.get("parameters", {})
        confirmations = tool_args.get("voice_confirmations", {})
        return await self.mcp_agent.execute_tool(user_id, tool_id, parameters, confirmations)

    async def _execute_mcp_configure_tool(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        user_id = tool_args.get("user_id")
        tool_id = tool_args.get("tool_id")
        enabled = bool(tool_args.get("enabled", True))
        priority = int(tool_args.get("priority", 5))
        custom_settings = tool_args.get("custom_settings", {})
        return await self.mcp_agent.configure_tool(user_id, tool_id, enabled, priority, custom_settings)

    async def _execute_mcp_setup_service_auth(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        user_id = tool_args.get("user_id")
        service_id = tool_args.get("service_id")
        auth_data = tool_args.get("auth_data", {})
        return await self.mcp_agent.setup_service_auth(user_id, service_id, auth_data)

    async def _execute_mcp_get_oauth_url(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        user_id = tool_args.get("user_id")
        service_id = tool_args.get("service_id")
        redirect_uri = tool_args.get("redirect_uri")
        return await self.mcp_agent.get_oauth_url(user_id, service_id, redirect_uri)

    async def _execute_mcp_get_user_profile(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        user_id = tool_args.get("user_id")
        return await self.mcp_agent.get_user_profile(user_id)

    async def _execute_mcp_authenticate(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        user_id = tool_args.get("user_id")
        auth_token = tool_args.get("auth_token")
        if not user_id or not auth_token:
            return {
                "success": False,
                "error": "user_id and auth_token are required for authentication"
            }
        return await self.mcp_agent.authenticate_user(user_id, auth_token)

    async def _execute_web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Execute a web search using Google Custom Search API and return structured data"""
        try:
            # Ensure num_results is an int and within allowed range
            try:
                num_results = int(num_results)
            except (ValueError, TypeError):
                num_results = 5
            num_results = max(1, min(num_results, 10))

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
                    "num": num_results
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

    async def _execute_create_reminder(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the create reminder tool"""
        try:
            reminder_service = await get_reminder_service()
            
            user_id = tool_args.get("user_id")
            title = tool_args.get("title")
            description = tool_args.get("description")
            reminder_time_str = tool_args.get("reminder_time")
            event_id = tool_args.get("event_id")
            
            if not all([user_id, title, description, reminder_time_str]):
                return {
                    "success": False, 
                    "error": "Missing required fields: user_id, title, description, reminder_time"
                }
            
            # Parse reminder time
            try:
                reminder_time = datetime.fromisoformat(reminder_time_str.replace('Z', '+00:00'))
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid reminder_time format. Please use ISO format (YYYY-MM-DDTHH:MM:SS)"
                }
            
            reminder_id = await reminder_service.create_reminder(
                user_id=user_id,
                title=title,
                description=description,
                reminder_time=reminder_time,
                event_id=event_id
            )
            
            return {
                "success": True, 
                "reminder_id": reminder_id,
                "message": f"Reminder '{title}' created successfully"
            }
            
        except Exception as e:
            return {
                "success": False, 
                "error": f"Error creating reminder: {str(e)}"
            }

    async def _execute_get_reminders(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the get user reminders tool"""
        try:
            reminder_service = await get_reminder_service()
            
            user_id = tool_args.get("user_id")
            status = tool_args.get("status", "active")
            limit = tool_args.get("limit", 10)
            
            if not user_id:
                return {
                    "success": False, 
                    "error": "User ID is required to get reminders"
                }
            
            # Get upcoming reminders if status is active
            if status == "active":
                reminders = await reminder_service.get_upcoming_reminders(user_id, limit)
            else:
                reminders = await reminder_service.get_user_reminders(user_id, status)
            
            return {
                "success": True, 
                "reminders": reminders,
                "count": len(reminders)
            }
            
        except Exception as e:
            return {
                "success": False, 
                "error": f"Error getting reminders: {str(e)}"
            }

    async def _execute_complete_reminder(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete reminder tool"""
        try:
            reminder_service = await get_reminder_service()
            
            reminder_id = tool_args.get("reminder_id")
            user_id = tool_args.get("user_id")
            
            if not all([reminder_id, user_id]):
                return {
                    "success": False, 
                    "error": "Both reminder_id and user_id are required"
                }
            
            success = await reminder_service.complete_reminder(reminder_id, user_id)
            
            if success:
                return {
                    "success": True, 
                    "message": f"Reminder completed successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to complete reminder - it may not exist or belong to this user"
                }
                
        except Exception as e:
            return {
                "success": False, 
                "error": f"Error completing reminder: {str(e)}"
            }

    async def _execute_delete_reminder(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the delete reminder tool"""
        try:
            reminder_service = await get_reminder_service()
            
            reminder_id = tool_args.get("reminder_id")
            user_id = tool_args.get("user_id")
            permission_granted = tool_args.get("permission_granted", False)
            
            if not all([reminder_id, user_id]):
                return {
                    "success": False, 
                    "error": "Both reminder_id and user_id are required"
                }
            
            # First check if permission is needed
            if not permission_granted:
                permission_request = await reminder_service.request_reminder_deletion(reminder_id, user_id)
                return permission_request
            
            # Delete with permission
            result = await reminder_service.delete_reminder_with_permission(
                reminder_id, user_id, permission_granted
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False, 
                "error": f"Error deleting reminder: {str(e)}"
            }

    async def _execute_login_greeting(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the handle login greeting tool"""
        try:
            greeting_service = await get_login_greeting_service()
            
            user_id = tool_args.get("user_id")
            session_id = tool_args.get("session_id")
            
            if not all([user_id, session_id]):
                return {
                    "success": False, 
                    "error": "Both user_id and session_id are required for login greeting"
                }
            
            greeting_result = await greeting_service.handle_user_login(user_id, session_id)
            
            return greeting_result
            
        except Exception as e:
            return {
                "success": False, 
                "error": f"Error handling login greeting: {str(e)}"
            }