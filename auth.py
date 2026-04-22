import os
import datetime
import jwt
import bcrypt
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from database import get_supabase, get_supabase_admin
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["auth"])

JWT_SECRET = os.getenv("JWT_SECRET", "fallback-secret")
ALGORITHM = "HS256"

class UserAuth(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

def hash_password(password: str):
    # Hash a password using bcrypt
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str):
    # Verify a password using bcrypt
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

@router.post("/signup")
async def signup(user: UserAuth):
    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    # 1. Check if user already exists
    try:
        existing_user = supabase.table("users").select("*").eq("email", user.email).execute()
        if existing_user.data:
            raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    # 2. Hash the password
    hashed_password = hash_password(user.password)
    
    # 3. Insert into custom users table
    try:
        new_user = {
            "email": user.email,
            "password": hashed_password,
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        response = supabase.table("users").insert(new_user).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to create user")
        
        user_id = response.data[0]["id"]
        
        # 4. Create initial profile entry
        if user.full_name:
            admin_supabase = get_supabase_admin()
            admin_supabase.table("profiles").insert({
                "id": user_id,
                "full_name": user.full_name
            }).execute()
            
        return {"message": "User created successfully", "user": {"email": user.email}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database error: {str(e)}")

@router.post("/signin")
async def signin(user: UserAuth):
    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    # 1. Fetch user by email
    try:
        response = supabase.table("users").select("*").eq("email", user.email).execute()
        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        db_user = response.data[0]
        
        # 2. Verify password
        if not verify_password(user.password, db_user["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # 3. Create JWT token
        token = create_access_token({"sub": user.email})
        
        return {
            "message": "Login successful",
            "session": {
                "access_token": token,
                "token_type": "bearer",
                "user": {"email": user.email}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
