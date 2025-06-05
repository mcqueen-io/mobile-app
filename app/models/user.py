from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from uuid import uuid4

class Relationship(BaseModel):
    user_id: str
    type: str
    since: datetime
    metadata: Optional[Dict] = Field(default_factory=dict)

class SchedulePreferences(BaseModel):
    preferred_times: List[str] = Field(default_factory=list)
    timezone: str = "UTC"

class CommunicationPreferences(BaseModel):
    preferred_language: str = "en"
    formality_level: int = Field(default=3, ge=1, le=5)
    notification_preferences: Dict = Field(default_factory=dict)

class UserPreferences(BaseModel):
    cuisine: List[str] = Field(default_factory=list)
    genre: List[str] = Field(default_factory=list)
    movies: List[str] = Field(default_factory=list)
    music: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    schedule: SchedulePreferences = Field(default_factory=SchedulePreferences)
    communication: CommunicationPreferences = Field(default_factory=CommunicationPreferences)

class VoiceProfile(BaseModel):
    embedding_version: str
    last_updated: datetime
    quality_score: float = Field(ge=0.0, le=1.0)

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    username: str
    voice_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    relationships: List[Relationship] = Field(default_factory=list)
    voice_profile: Optional[VoiceProfile] = None
    family_tree_id: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> dict:
        """Convert user model to dictionary"""
        return self.dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Create user model from dictionary"""
        return cls(**data)

    def update_preferences(self, new_preferences: Dict) -> None:
        """Update user preferences"""
        current_prefs = self.preferences.dict()
        current_prefs.update(new_preferences)
        self.preferences = UserPreferences(**current_prefs)
        self.updated_at = datetime.utcnow()

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a new relationship"""
        self.relationships.append(relationship)
        self.updated_at = datetime.utcnow()

    def update_voice_profile(self, embedding_version: str, quality_score: float) -> None:
        """Update voice profile information"""
        self.voice_profile = VoiceProfile(
            embedding_version=embedding_version,
            last_updated=datetime.utcnow(),
            quality_score=quality_score
        )
        self.updated_at = datetime.utcnow() 