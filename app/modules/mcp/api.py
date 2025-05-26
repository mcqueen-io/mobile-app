from fastapi import APIRouter, HTTPException
from app.modules.mcp.mcp_client import get_mcp_client
from typing import Dict, List
from pydantic import BaseModel

router = APIRouter()

class ToolRequest(BaseModel):
    parameters: Dict

@router.get("/tools")
async def get_available_tools():
    """Get list of available tools"""
    try:
        mcp_client = get_mcp_client()
        tools = await mcp_client.get_available_tools()
        return tools
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: ToolRequest):
    """Execute a specific tool"""
    try:
        mcp_client = get_mcp_client()
        result = await mcp_client.execute_tool(tool_name, request.parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/maps/{action}")
async def execute_map_tool(action: str, parameters: Dict):
    """Execute a map-related tool"""
    try:
        mcp_client = get_mcp_client()
        result = await mcp_client.execute_map_tool(action, parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/email/{action}")
async def execute_email_tool(action: str, parameters: Dict):
    """Execute an email-related tool"""
    try:
        mcp_client = get_mcp_client()
        result = await mcp_client.execute_email_tool(action, parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/calendar/{action}")
async def execute_calendar_tool(action: str, parameters: Dict):
    """Execute a calendar-related tool"""
    try:
        mcp_client = get_mcp_client()
        result = await mcp_client.execute_calendar_tool(action, parameters)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 