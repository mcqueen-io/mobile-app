from typing import Any, Dict, Optional
from app.modules.mcp.mcp_client import get_mcp_client


class MCPChildAgent:
    """Child agent that encapsulates MCP server interactions for tools/actions."""

    def __init__(self):
        self.client = get_mcp_client()

    async def authenticate_user(self, user_id: str, auth_token: str) -> Dict[str, Any]:
        resp = await self.client.authenticate_user(user_id, auth_token)
        return resp.model_dump()

    async def list_tools(self, user_id: str) -> Dict[str, Any]:
        resp = await self.client.get_available_tools(user_id)
        return resp.model_dump()

    async def execute_tool(self, user_id: str, tool_id: str, parameters: Dict[str, Any], voice_confirmations: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        resp = await self.client.execute_tool(user_id, tool_id, parameters, voice_confirmations)
        return resp.model_dump()

    async def configure_tool(self, user_id: str, tool_id: str, enabled: bool = True, priority: int = 5, custom_settings: Optional[Dict] = None) -> Dict[str, Any]:
        resp = await self.client.configure_tool(user_id, tool_id, enabled, priority, custom_settings)
        return resp.model_dump()

    async def setup_service_auth(self, user_id: str, service_id: str, auth_data: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self.client.setup_service_auth(user_id, service_id, auth_data)
        return resp.model_dump()

    async def get_oauth_url(self, user_id: str, service_id: str, redirect_uri: str) -> Dict[str, Any]:
        resp = await self.client.get_oauth_url(user_id, service_id, redirect_uri)
        return resp.model_dump()

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        resp = await self.client.get_user_profile(user_id)
        return resp.model_dump()

    async def logout_user(self, user_id: str) -> Dict[str, Any]:
        resp = await self.client.logout_user(user_id)
        return resp.model_dump()


