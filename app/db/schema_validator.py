from typing import Dict, Any
import json
from datetime import datetime

class SchemaValidator:
    @staticmethod
    def validate_user(user_data: Dict[str, Any]) -> bool:
        """Validate user data structure"""
        required_fields = {'name', 'voice_id'}
        if not all(field in user_data for field in required_fields):
            return False

        # Validate preferences structure if present
        if 'preferences' in user_data:
            if not isinstance(user_data['preferences'], dict):
                return False

        # Validate relationships if present
        if 'relationships' in user_data:
            if not isinstance(user_data['relationships'], list):
                return False
            for rel in user_data['relationships']:
                if not all(k in rel for k in ['user_id', 'type', 'since']):
                    return False

        return True

    @staticmethod
    def validate_event(event_data: Dict[str, Any]) -> bool:
        """Validate event data structure"""
        required_fields = {
            'type', 'title', 'description', 'datetime',
            'participants', 'created_by', 'status', 'importance'
        }
        if not all(field in event_data for field in required_fields):
            return False

        # Validate datetime
        if not isinstance(event_data['datetime'], datetime):
            return False

        # Validate participants
        if not isinstance(event_data['participants'], list):
            return False

        # Validate status
        valid_statuses = {'UPCOMING', 'COMPLETED', 'CANCELLED'}
        if event_data['status'] not in valid_statuses:
            return False

        # Validate importance
        if not isinstance(event_data['importance'], int) or \
           not 1 <= event_data['importance'] <= 5:
            return False

        return True

    @staticmethod
    def validate_preferences(preferences: Dict[str, Any]) -> bool:
        """Validate preferences structure"""
        if not isinstance(preferences, dict):
            return False

        # Validate each preference category
        for category, value in preferences.items():
            if category == 'schedule':
                if not isinstance(value, dict):
                    return False
                if 'preferred_times' in value and not isinstance(value['preferred_times'], list):
                    return False
                if 'timezone' in value and not isinstance(value['timezone'], str):
                    return False
            elif category == 'communication':
                if not isinstance(value, dict):
                    return False
                if 'formality_level' in value and not isinstance(value['formality_level'], int):
                    return False
            else:
                # For other categories, ensure they're lists
                if not isinstance(value, list):
                    return False

        return True

    @staticmethod
    def validate_relationship(relationship: Dict[str, Any]) -> bool:
        """Validate relationship data structure"""
        required_fields = {'user_id', 'type', 'since'}
        if not all(field in relationship for field in required_fields):
            return False

        # Validate relationship type
        valid_types = {
            'PARENT_OF', 'CHILD_OF', 'SPOUSE_OF', 'SIBLING_OF',
            'CONNECTED_TO', 'FRIEND_OF', 'COLLEAGUE_OF'
        }
        if relationship['type'] not in valid_types:
            return False

        # Validate since date
        if not isinstance(relationship['since'], datetime):
            return False

        # Validate metadata if present
        if 'metadata' in relationship:
            if not isinstance(relationship['metadata'], dict):
                return False
            if 'relationship_strength' in relationship['metadata']:
                strength = relationship['metadata']['relationship_strength']
                if not isinstance(strength, int) or not 1 <= strength <= 5:
                    return False

        return True

# Create singleton instance
schema_validator = SchemaValidator()

def get_schema_validator():
    return schema_validator 