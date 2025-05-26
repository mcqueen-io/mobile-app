from neo4j import GraphDatabase
from app.core.config import settings
from typing import Dict, List, Optional
import json

class UserGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def create_user(self, user_id: str, properties: Dict) -> None:
        """Create a new user node"""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (u:User {
                    id: $user_id,
                    name: $name,
                    preferences: $preferences,
                    voice_embedding: $voice_embedding
                })
                """,
                user_id=user_id,
                name=properties.get('name'),
                preferences=json.dumps(properties.get('preferences', {})),
                voice_embedding=properties.get('voice_embedding')
            )

    def create_relationship(self, user1_id: str, user2_id: str, relationship_type: str) -> None:
        """Create a relationship between two users"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (u1:User {id: $user1_id})
                MATCH (u2:User {id: $user2_id})
                CREATE (u1)-[r:RELATES_TO {type: $relationship_type}]->(u2)
                """,
                user1_id=user1_id,
                user2_id=user2_id,
                relationship_type=relationship_type
            )

    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})
                RETURN u.preferences as preferences
                """,
                user_id=user_id
            )
            record = result.single()
            return json.loads(record["preferences"]) if record else {}

    def update_user_preferences(self, user_id: str, preferences: Dict) -> None:
        """Update user preferences"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (u:User {id: $user_id})
                SET u.preferences = $preferences
                """,
                user_id=user_id,
                preferences=json.dumps(preferences)
            )

    def get_family_tree(self, user_id: str) -> List[Dict]:
        """Get the family tree for a user"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})
                CALL {
                    WITH u
                    MATCH (u)-[r:RELATES_TO*1..3]-(related:User)
                    RETURN collect({
                        id: related.id,
                        name: related.name,
                        relationship: r[0].type
                    }) as family
                }
                RETURN family
                """,
                user_id=user_id
            )
            record = result.single()
            return record["family"] if record else []

# Create a singleton instance
user_graph = UserGraph()

def get_user_graph():
    return user_graph 