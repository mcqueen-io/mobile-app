import aiohttp
from app.core.config import settings
from typing import Dict, List, Optional
import json

class MCPClient:
    def __init__(self):
        self.base_url = settings.MCP_SERVER_URL
        self.session = None

    async def get_session(self):
        """Get or create an aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a tool through MCP"""
        session = await self.get_session()
        async with session.post(
            f"{self.base_url}/tools/{tool_name}",
            json=parameters
        ) as response:
            return await response.json()

    async def get_available_tools(self) -> List[Dict]:
        """Get list of available tools"""
        session = await self.get_session()
        async with session.get(f"{self.base_url}/tools") as response:
            return await response.json()

    async def execute_map_tool(self, action: str, parameters: Dict) -> Dict:
        """Execute map-related tool"""
        return await self.execute_tool("maps", {
            "action": action,
            **parameters
        })

    async def execute_email_tool(self, action: str, parameters: Dict) -> Dict:
        """Execute email-related tool"""
        return await self.execute_tool("email", {
            "action": action,
            **parameters
        })

    async def execute_calendar_tool(self, action: str, parameters: Dict) -> Dict:
        """Execute calendar-related tool"""
        return await self.execute_tool("calendar", {
            "action": action,
            **parameters
        })

# Create a singleton instance
mcp_client = MCPClient()

def get_mcp_client():
    return mcp_client 