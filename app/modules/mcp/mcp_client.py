import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from app.core.config import settings
from pydantic import BaseModel
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPResponse(BaseModel):
    """Standard response from MCP server"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class MCPSession(BaseModel):
    """MCP session information"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime

class MCPClient:
    """Client for communicating with the MCP Server microservice"""
    
    def __init__(self):
        self.base_url = settings.MCP_SERVER_URL
        self.session: Optional[aiohttp.ClientSession] = None
        self.user_sessions: Dict[str, MCPSession] = {}
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def authenticate_user(self, user_id: str, auth_token: str) -> MCPResponse:
        """Authenticate user with MCP server and create session"""
        try:
            session = await self.get_session()
            
            payload = {
                "user_id": user_id,
                "auth_token": auth_token,
                "client_type": "mobile_app"
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/auth/login",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    # Store session info
                    session_data = result.get("session", {})
                    self.user_sessions[user_id] = MCPSession(
                        session_id=session_data["session_id"],
                        user_id=user_id,
                        created_at=datetime.fromisoformat(session_data["created_at"]),
                        expires_at=datetime.fromisoformat(session_data["expires_at"])
                    )
                    
                    return MCPResponse(
                        success=True,
                        data=session_data,
                        metadata={"authenticated": True}
                    )
                else:
                    return MCPResponse(
                        success=False,
                        error=f"Authentication failed: {result.get('error', 'Unknown error')}"
                    )
                    
        except Exception as e:
            logger.error(f"Error authenticating user {user_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"Authentication error: {str(e)}"
            )
    
    async def get_available_tools(self, user_id: str) -> MCPResponse:
        """Get available tools for a user"""
        try:
            headers = await self._get_auth_headers(user_id)
            if not headers:
                return MCPResponse(success=False, error="User not authenticated")
            
            session = await self.get_session()
            
            async with session.get(
                f"{self.base_url}/api/v1/mcp/tools",
                headers=headers,
                params={"user_id": user_id}
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    return MCPResponse(
                        success=True,
                        data=result.get("tools", []),
                        metadata={"total": result.get("total", 0)}
                    )
                else:
                    return MCPResponse(
                        success=False,
                        error=f"Failed to get tools: {result.get('error', 'Unknown error')}"
                    )
                    
        except Exception as e:
            logger.error(f"Error getting tools for user {user_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"Get tools error: {str(e)}"
            )
    
    async def execute_tool(self, user_id: str, tool_id: str, parameters: Dict[str, Any], 
                          voice_confirmations: Optional[Dict[str, bool]] = None) -> MCPResponse:
        """Execute a tool on the MCP server"""
        try:
            headers = await self._get_auth_headers(user_id)
            if not headers:
                return MCPResponse(success=False, error="User not authenticated")
            
            session = await self.get_session()
            
            payload = {
                "parameters": parameters,
                "voice_confirmations": voice_confirmations or {},
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/mcp/tools/{tool_id}/execute",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                return MCPResponse(
                    success=result.get("success", False),
                    data=result.get("data"),
                    error=result.get("error"),
                    metadata=result.get("metadata", {})
                )
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_id} for user {user_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"Tool execution error: {str(e)}"
            )
    
    async def configure_tool(self, user_id: str, tool_id: str, 
                           enabled: bool = True, priority: int = 5, 
                           custom_settings: Optional[Dict] = None) -> MCPResponse:
        """Configure a tool for a user"""
        try:
            headers = await self._get_auth_headers(user_id)
            if not headers:
                return MCPResponse(success=False, error="User not authenticated")
            
            session = await self.get_session()
            
            payload = {
                "tool_id": tool_id,
                "enabled": enabled,
                "priority": priority,
                "custom_settings": custom_settings or {}
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/mcp/tools/{tool_id}/configure",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                return MCPResponse(
                    success=result.get("success", False),
                    data=result.get("data"),
                    error=result.get("error")
                )
                
        except Exception as e:
            logger.error(f"Error configuring tool {tool_id} for user {user_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"Tool configuration error: {str(e)}"
            )
    
    async def setup_service_auth(self, user_id: str, service_id: str, 
                               auth_data: Dict[str, Any]) -> MCPResponse:
        """Set up authentication for a service"""
        try:
            headers = await self._get_auth_headers(user_id)
            if not headers:
                return MCPResponse(success=False, error="User not authenticated")
            
            session = await self.get_session()
            
            payload = {
                "service_id": service_id,
                "auth_data": auth_data
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/mcp/auth/{service_id}/setup",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                return MCPResponse(
                    success=result.get("success", False),
                    data=result.get("data"),
                    error=result.get("error")
                )
                
        except Exception as e:
            logger.error(f"Error setting up auth for {service_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"Auth setup error: {str(e)}"
            )
    
    async def get_oauth_url(self, user_id: str, service_id: str, 
                          redirect_uri: str) -> MCPResponse:
        """Get OAuth URL for a service"""
        try:
            headers = await self._get_auth_headers(user_id)
            if not headers:
                return MCPResponse(success=False, error="User not authenticated")
            
        session = await self.get_session()
            
            payload = {
                "service_id": service_id,
                "redirect_uri": redirect_uri
            }
            
        async with session.post(
                f"{self.base_url}/api/v1/mcp/auth/{service_id}/oauth-url",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                return MCPResponse(
                    success=result.get("success", False),
                    data=result.get("data"),
                    error=result.get("error")
                )
                
        except Exception as e:
            logger.error(f"Error getting OAuth URL for {service_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"OAuth URL error: {str(e)}"
            )
    
    async def get_user_profile(self, user_id: str) -> MCPResponse:
        """Get user's MCP profile"""
        try:
            headers = await self._get_auth_headers(user_id)
            if not headers:
                return MCPResponse(success=False, error="User not authenticated")
            
            session = await self.get_session()
            
            async with session.get(
                f"{self.base_url}/api/v1/mcp/users/{user_id}/profile",
                headers=headers
            ) as response:
                result = await response.json()
                
                return MCPResponse(
                    success=result.get("success", False),
                    data=result.get("data"),
                    error=result.get("error")
                )
                
        except Exception as e:
            logger.error(f"Error getting profile for user {user_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"Profile error: {str(e)}"
            )
    
    async def health_check(self) -> MCPResponse:
        """Check MCP server health"""
        try:
            session = await self.get_session()
            
            async with session.get(
                f"{self.base_url}/api/v1/health"
        ) as response:
                result = await response.json()
                
                return MCPResponse(
                    success=response.status == 200,
                    data=result,
                    metadata={"status_code": response.status}
                )
                
        except Exception as e:
            logger.error(f"Error checking MCP server health: {e}")
            return MCPResponse(
                success=False,
                error=f"Health check error: {str(e)}"
            )
    
    async def _get_auth_headers(self, user_id: str) -> Optional[Dict[str, str]]:
        """Get authentication headers for requests"""
        if user_id not in self.user_sessions:
            return None
        
        session_info = self.user_sessions[user_id]
        
        # Check if session is expired
        if datetime.utcnow() > session_info.expires_at:
            del self.user_sessions[user_id]
            return None
        
        return {
            "Authorization": f"Bearer {session_info.session_id}",
            "X-User-ID": user_id,
            "X-Session-ID": session_info.session_id,
            "Content-Type": "application/json"
        }
    
    def is_user_authenticated(self, user_id: str) -> bool:
        """Check if user has valid session"""
        if user_id not in self.user_sessions:
            return False
        
        session_info = self.user_sessions[user_id]
        return datetime.utcnow() <= session_info.expires_at
    
    async def logout_user(self, user_id: str) -> MCPResponse:
        """Logout user and cleanup session"""
        try:
            if user_id in self.user_sessions:
                headers = await self._get_auth_headers(user_id)
                if headers:
        session = await self.get_session()
                    
                    # Notify server of logout
                    async with session.post(
                        f"{self.base_url}/api/v1/auth/logout",
                        headers=headers
                    ) as response:
                        pass  # Don't care about response
                
                # Remove local session
                del self.user_sessions[user_id]
            
            return MCPResponse(success=True, data={"logged_out": True})
            
        except Exception as e:
            logger.error(f"Error logging out user {user_id}: {e}")
            return MCPResponse(
                success=False,
                error=f"Logout error: {str(e)}"
            )

# Global instance
mcp_client = MCPClient()

def get_mcp_client() -> MCPClient:
    """Get the global MCP client instance"""
    return mcp_client 