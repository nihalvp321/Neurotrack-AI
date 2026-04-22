from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Profile(Base):
    __tablename__ = "profiles"
    
    # Referencing public.users(id) which is Integer
    id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    full_name = Column(Text)
    age = Column(Integer)
    gender = Column(String)
    avatar_url = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    
    user = relationship("User", backref="profile")

class TestSession(Base):
    __tablename__ = "test_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("profiles.id"))
    tremor_score = Column(Float)
    voice_score = Column(Float)
    spiral_score = Column(Float)
    gait_score = Column(Float)
    tapping_score = Column(Float)
    final_risk = Column(Float)
    severity_stage = Column(Text)
    explanation = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    profile = relationship("Profile", backref="sessions")
