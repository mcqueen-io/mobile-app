import asyncio
import logging
from datetime import datetime
from app.services.user_service import get_user_service
from app.db.mongo_manager import get_mongo_manager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_test_users():
    """Create test users with different preferences"""
    service = await get_user_service()
    
    # Create main user
    main_user = await service.create_user({
        "name": "John Doe",
        "voice_id": "voice_123",
        "preferences": {
            "cuisine": ["Italian", "Japanese", "Mexican"],
            "genre": ["Rock", "Jazz"],
            "activities": ["Hiking", "Swimming"],
            "schedule": {
                "preferred_times": ["morning", "evening"],
                "timezone": "UTC+1"
            }
        }
    })
    logger.info(f"Created main user: {main_user.name}")

    # Create related users
    users_data = [
        {
            "name": "Alice Smith",
            "voice_id": "voice_456",
            "preferences": {
                "cuisine": ["Italian", "Japanese", "Thai"],
                "genre": ["Jazz", "Classical"],
                "activities": ["Swimming", "Yoga"]
            }
        },
        {
            "name": "Bob Johnson",
            "voice_id": "voice_789",
            "preferences": {
                "cuisine": ["Mexican", "Indian"],
                "genre": ["Rock", "Blues"],
                "activities": ["Hiking", "Cycling"]
            }
        },
        {
            "name": "Carol White",
            "voice_id": "voice_101",
            "preferences": {
                "cuisine": ["Italian", "French"],
                "genre": ["Classical", "Jazz"],
                "activities": ["Yoga", "Swimming"]
            }
        }
    ]

    created_users = []
    for user_data in users_data:
        user = await service.create_user(user_data)
        created_users.append(user)
        logger.info(f"Created user: {user.name}")

    return main_user, created_users

async def establish_relationships(main_user, other_users):
    """Establish relationships between users"""
    service = await get_user_service()
    
    # Add direct relationships
    await service.add_relationship(main_user.id, {
        "user_id": other_users[0].id,
        "type": "FRIEND_OF",
        "since": datetime.utcnow(),
        "metadata": {"relationship_strength": 4}
    })
    logger.info(f"Added relationship between {main_user.name} and {other_users[0].name}")

    # Add relationship between Alice and Carol
    await service.add_relationship(other_users[0].id, {
        "user_id": other_users[2].id,
        "type": "FRIEND_OF",
        "since": datetime.utcnow(),
        "metadata": {"relationship_strength": 3}
    })
    logger.info(f"Added relationship between {other_users[0].name} and {other_users[2].name}")

async def test_connection_suggestions(main_user):
    """Test connection suggestions"""
    service = await get_user_service()
    
    # Get connection suggestions
    suggestions = await service.suggest_connections(
        main_user.id,
        max_depth=2,
        min_common_interests=1
    )
    
    logger.info("\nConnection Suggestions:")
    for suggestion in suggestions:
        user = suggestion['suggested_user']
        logger.info(f"- {user['name']}:")
        logger.info(f"  Depth: {suggestion['depth']}")
        logger.info(f"  Common interests: {suggestion['common_interests']}")

async def test_preference_updates(main_user):
    """Test preference updates and history"""
    service = await get_user_service()
    
    # Update preferences
    new_preferences = {
        "cuisine": ["Italian", "Japanese", "Mexican", "Thai"],
        "activities": ["Hiking", "Swimming", "Cycling"]
    }
    
    await service.update_preferences(main_user.id, new_preferences)
    logger.info(f"\nUpdated preferences for {main_user.name}")
    
    # Get preference history
    history = await service.get_preference_history(main_user.id)
    logger.info("\nPreference History:")
    for entry in history:
        logger.info(f"- {entry['preference_type']}:")
        logger.info(f"  Old: {entry['old_value']}")
        logger.info(f"  New: {entry['new_value']}")
        logger.info(f"  Time: {entry['timestamp']}")

async def main():
    """Main test function"""
    try:
        # Initialize MongoDB
        mongo_manager = await get_mongo_manager()
        await mongo_manager.create_collections()
        
        # Create test users
        main_user, other_users = await create_test_users()
        
        # Establish relationships
        await establish_relationships(main_user, other_users)
        
        # Test connection suggestions
        await test_connection_suggestions(main_user)
        
        # Test preference updates and history
        await test_preference_updates(main_user)
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
    finally:
        # Clean up
        # Ensure mongo_manager was successfully initialized before attempting to close
        if 'mongo_manager' in locals() and mongo_manager is not None:
             await mongo_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 