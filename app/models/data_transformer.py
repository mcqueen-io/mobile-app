from typing import Dict, Any, Optional, List
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel

class DataTransformer:
    """
    Centralized data transformation layer.
    Handles all transformations between MongoDB, Pydantic models, and API responses.
    """
    
    @staticmethod
    def mongo_to_pydantic(data: Dict[str, Any], model_class: type[BaseModel]) -> BaseModel:
        """Transform MongoDB document to Pydantic model"""
        if not data:
            return None
            
        # Convert ObjectId to string
        if '_id' in data:
            data['id'] = str(data.pop('_id'))
            
        # Handle nested ObjectIds
        for key, value in data.items():
            if isinstance(value, dict) and '_id' in value:
                value['id'] = str(value.pop('_id'))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and '_id' in item:
                        item['id'] = str(item.pop('_id'))
        
        return model_class.model_validate(data)

    @staticmethod
    def pydantic_to_mongo(model: BaseModel) -> Dict[str, Any]:
        """Transform Pydantic model to MongoDB document"""
        if not model:
            return None
            
        data = model.model_dump()
        
        # Convert string IDs back to ObjectId
        if 'id' in data:
            data['_id'] = ObjectId(data.pop('id'))
            
        # Handle nested IDs
        for key, value in data.items():
            if isinstance(value, dict) and 'id' in value:
                value['_id'] = ObjectId(value.pop('id'))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and 'id' in item:
                        item['_id'] = ObjectId(item.pop('id'))
        
        return data

    @staticmethod
    def format_datetime(dt: datetime) -> str:
        """Format datetime for API responses"""
        return dt.isoformat() if dt else None

    @staticmethod
    async def format_memory(memory: Dict[str, Any], username_resolver) -> Dict[str, Any]:
        """Format memory for API response"""
        created_by_username = await username_resolver(memory.get('created_by'))
        return {
            "memory_id": memory.get('memory_id'),
            "content": memory.get('content', ''),
            "created_by": created_by_username,
            "created_at": DataTransformer.format_datetime(memory.get('created_at')),
            "type": memory.get('type', 'unknown')
        }

    @staticmethod
    def format_user_context(user) -> Dict[str, Any]:
        """Format user data for context, handling both dicts and Pydantic models"""
        if isinstance(user, dict):
            user_id = str(user.get('id', user.get('_id', '')))
            username = user.get('username', '')
            name = user.get('name', '')
            preferences = user.get('preferences', {})
            # If preferences is a Pydantic model, convert to dict
            if hasattr(preferences, 'model_dump'):
                preferences = preferences.model_dump()
            preferences = {k: v for k, v in preferences.items() if v is not None}
        else:
            user_id = str(user.id)
            username = user.username
            name = user.name
            preferences = {k: v for k, v in user.preferences.model_dump().items() if v is not None}
        return {
            "user_id": user_id,
            "username": username,
            "name": name,
            "preferences": preferences
        }

    @staticmethod
    def format_memory_search_results(memories: List[Dict[str, Any]], username_resolver) -> str:
        """Format memory search results as a string"""
        return "\n".join([
            f"- [{m.get('type', 'unknown').upper()}] {m.get('content', 'No content')} "
            f"(Created by: {username_resolver(m.get('created_by'))}, "
            f"Time: {DataTransformer.format_datetime(m.get('created_at'))})"
            for m in memories
        ]) 