from fastapi import APIRouter, HTTPException, Depends
from app.modules.mcp.mcp_client import get_mcp_client, MCPResponse
from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class ToolRequest(BaseModel):
    parameters: Dict
    voice_confirmations: Optional[Dict[str, bool]] = None

class AuthSetupRequest(BaseModel):
    service_id: str
    auth_data: Dict

class ToolConfigRequest(BaseModel):
    tool_id: str
    enabled: bool = True
    priority: int = 5
    custom_settings: Optional[Dict] = None

class OAuthRequest(BaseModel):
    service_id: str
    redirect_uri: str

class AuthRequest(BaseModel):
    user_id: str
    auth_token: str

# Dependency to get current user (placeholder - implement based on your auth system)
async def get_current_user_id() -> str:
    # This should be implemented based on your authentication system
    # For now, return a mock user ID
    return "test_user_123"

@router.post("/auth/login")
async def authenticate_user(request: AuthRequest):
    """Authenticate user with MCP server"""
    try:
        mcp_client = get_mcp_client()
        result = await mcp_client.authenticate_user(request.user_id, request.auth_token)
        
        if result.success:
            return {
                "success": True,
                "session": result.data,
                "message": "User authenticated successfully"
            }
        else:
            raise HTTPException(status_code=401, detail=result.error)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools")
async def get_available_tools(user_id: str = Depends(get_current_user_id)):
    """Get list of available tools for the current user"""
    try:
        mcp_client = get_mcp_client()
        
        # Check if user is authenticated
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated with MCP server")
        
        result = await mcp_client.get_available_tools(user_id)
        
        if result.success:
            return {
                "tools": result.data,
                "total": result.metadata.get("total", 0),
                "success": True
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/{tool_id}/execute")
async def execute_tool(tool_id: str, request: ToolRequest, user_id: str = Depends(get_current_user_id)):
    """Execute a specific tool for the user"""
    try:
        mcp_client = get_mcp_client()
        
        # Check if user is authenticated
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated with MCP server")
        
        result = await mcp_client.execute_tool(
            user_id=user_id,
            tool_id=tool_id,
            parameters=request.parameters,
            voice_confirmations=request.voice_confirmations
        )
        
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/{tool_id}/configure")
async def configure_tool(tool_id: str, request: ToolConfigRequest, user_id: str = Depends(get_current_user_id)):
    """Configure a tool for the user"""
    try:
        mcp_client = get_mcp_client()
        
        # Check if user is authenticated
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated with MCP server")
        
        result = await mcp_client.configure_tool(
            user_id=user_id,
            tool_id=tool_id,
            enabled=request.enabled,
            priority=request.priority,
            custom_settings=request.custom_settings
        )
        
        if result.success:
            return {"success": True, "data": result.data}
        else:
            raise HTTPException(status_code=400, detail=result.error)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/setup")
async def setup_authentication(request: AuthSetupRequest, user_id: str = Depends(get_current_user_id)):
    """Set up authentication for a service"""
    try:
        mcp_client = get_mcp_client()
        
        # Check if user is authenticated
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated with MCP server")
        
        result = await mcp_client.setup_service_auth(
            user_id=user_id,
            service_id=request.service_id,
            auth_data=request.auth_data
        )
        
        if result.success:
            return {"success": True, "data": result.data}
        else:
            raise HTTPException(status_code=400, detail=result.error)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/oauth-url")
async def get_oauth_url(request: OAuthRequest, user_id: str = Depends(get_current_user_id)):
    """Get OAuth authorization URL for a service"""
    try:
        mcp_client = get_mcp_client()
        
        # Check if user is authenticated
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated with MCP server")
        
        result = await mcp_client.get_oauth_url(
            user_id=user_id,
            service_id=request.service_id,
            redirect_uri=request.redirect_uri
        )
        
        if result.success:
            return {"oauth_url": result.data.get("oauth_url")}
        else:
            raise HTTPException(status_code=400, detail=result.error)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/profile")
async def get_user_mcp_profile(user_id: str = Depends(get_current_user_id)):
    """Get user's MCP profile and configurations"""
    try:
        mcp_client = get_mcp_client()
        
        # Check if user is authenticated
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated with MCP server")
        
        result = await mcp_client.get_user_profile(user_id)
        
        if result.success:
            return {
                "user_id": user_id,
                "profile": result.data,
                "success": True
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Get health status of MCP server"""
    try:
        mcp_client = get_mcp_client()
        result = await mcp_client.health_check()
        
        return {
            "mcp_server_status": "healthy" if result.success else "unhealthy",
            "server_data": result.data,
            "client_status": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "mcp_server_status": "unreachable",
            "client_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/auth/logout")
async def logout_user(user_id: str = Depends(get_current_user_id)):
    """Logout user from MCP server"""
    try:
        mcp_client = get_mcp_client()
        result = await mcp_client.logout_user(user_id)
        
        return {
            "success": result.success,
            "message": "User logged out successfully" if result.success else result.error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility - now proxy to server
@router.post("/tools/maps/{action}")
async def execute_map_tool(action: str, parameters: Dict, user_id: str = Depends(get_current_user_id)):
    """Execute a map-related tool (legacy endpoint - proxies to server)"""
    try:
        mcp_client = get_mcp_client()
        
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Map legacy action to new tool ID
        tool_id = "maps_hybrid"
        mapped_params = {"action": action, **parameters}
        
        result = await mcp_client.execute_tool(
            user_id=user_id,
            tool_id=tool_id,
            parameters=mapped_params
        )
        
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/email/{action}")
async def execute_email_tool(action: str, parameters: Dict, user_id: str = Depends(get_current_user_id)):
    """Execute an email-related tool (legacy endpoint - proxies to server)"""
    try:
        mcp_client = get_mcp_client()
        
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Map legacy action to new tool ID
        tool_id = "email_personal"
        mapped_params = {"action": action, **parameters}
        
        result = await mcp_client.execute_tool(
            user_id=user_id,
            tool_id=tool_id,
            parameters=mapped_params
        )
        
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/calendar/{action}")
async def execute_calendar_tool(action: str, parameters: Dict, user_id: str = Depends(get_current_user_id)):
    """Execute a calendar-related tool (legacy endpoint - proxies to server)"""
    try:
        mcp_client = get_mcp_client()
        
        if not mcp_client.is_user_authenticated(user_id):
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Map legacy action to new tool ID
        tool_id = "calendar_personal"
        mapped_params = {"action": action, **parameters}
        
        result = await mcp_client.execute_tool(
            user_id=user_id,
            tool_id=tool_id,
            parameters=mapped_params
        )
        
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 