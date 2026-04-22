from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import auth
import os
from database import get_supabase, get_supabase_admin

app = FastAPI(title="Neurotrack AI API")

# Configure CORS for Flutter (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)

class ProfileUpdate(BaseModel):
    full_name: str
    age: int
    gender: str
    avatar_url: Optional[str] = None

@app.post("/profile/upload")
async def upload_profile_pic(email: str, file: UploadFile = File(...)):
    supabase = get_supabase_admin()  # Use admin client to bypass RLS for storage
    file_content = await file.read()
    
    # Use a consistent filename per user to avoid duplicates
    extension = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    file_path = f"{email}/avatar.{extension}"
    
    try:
        # Remove old avatar first (ignore error if it doesn't exist)
        try:
            supabase.storage.from_("profiles").remove([file_path])
        except Exception:
            pass
        
        # Upload to Supabase Storage with upsert enabled
        supabase.storage.from_("profiles").upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type, "upsert": "true"}
        )
        
        # Get Public URL
        url_res = supabase.storage.from_("profiles").get_public_url(file_path)
        
        # Save URL to profiles table
        user_res = supabase.table("users").select("id").eq("email", email).execute()
        if user_res.data:
            user_id = user_res.data[0]["id"]
            supabase.table("profiles").upsert({
                "id": user_id,
                "avatar_url": url_res
            }).execute()
            
        return {"url": url_res}
    except Exception as e:
        print(f"[Upload Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profile/update")
async def update_profile(email: str, profile: ProfileUpdate):
    supabase = get_supabase()
    user_res = supabase.table("users").select("id").eq("email", email).execute()
    if not user_res.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user_res.data[0]["id"]
    
    update_data = {
        "id": user_id,
        "full_name": profile.full_name,
        "age": profile.age,
        "gender": profile.gender
    }
    if profile.avatar_url:
        update_data["avatar_url"] = profile.avatar_url
        
    res = supabase.table("profiles").upsert(update_data).execute()
    
    return {"message": "Profile updated successfully"}

@app.get("/profile")
async def get_profile(email: str):
    supabase = get_supabase()
    user_res = supabase.table("users").select("id").eq("email", email).execute()
    if not user_res.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user_res.data[0]["id"]
    
    profile_res = supabase.table("profiles").select("*").eq("id", user_id).execute()
    if not profile_res.data:
        return {"full_name": "New User", "id": user_id}
    return profile_res.data[0]

@app.get("/sessions/latest")
async def get_latest_session(email: str):
    supabase = get_supabase()
    user_res = supabase.table("users").select("id").eq("email", email).execute()
    if not user_res.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user_res.data[0]["id"]
    
    session_res = supabase.table("test_sessions")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .limit(1)\
        .execute()
        
    if not session_res.data:
        return None
    return session_res.data[0]

@app.post("/predict")
async def predict(data: dict):
    # Calculate overall risk weighted average
    # healthy = 0.0, higher risk = 1.0
    tremor = data.get('tremor_score', 0)
    voice = data.get('voice_score', 0)
    spiral = data.get('spiral_score', 0)
    gait = data.get('gait_score', 0)
    tapping = data.get('tapping_score', 0)
    
    risk_score = (
        tremor * 0.25 +
        voice * 0.20 +
        spiral * 0.10 +
        gait * 0.25 +
        tapping * 0.20
    ) * 100
    
    severity = "Low"
    if risk_score > 60:
        severity = "High"
    elif risk_score > 30:
        severity = "Moderate"
        
    explanation = f"Your assessment shows a {severity.lower()} clinical risk profile. "
    if tremor > 0.5:
        explanation += "Elevated micro-tremors were detected during the stillness test. "
    if gait > 0.5:
        explanation += "We observed higher stride variability which may indicate balance instability. "
    if tapping < 0.4:
        explanation += "Finger tapping speed and rhythm are within the normal range. "
    else:
        explanation += "Reduced tapping cadence was noted. "
        
    explanation += "\n\nNote: This is an AI-assisted screening tool, not a diagnosis. Please consult a neurologist for a formal evaluation."
    
    # Store session in DB
    supabase = get_supabase_admin()
    user_res = supabase.table("users").select("id").eq("email", data.get('user_id')).execute()
    if user_res.data:
        user_id = user_res.data[0]["id"]
        supabase.table("test_sessions").insert({
            "user_id": user_id,
            "tremor_score": tremor,
            "voice_score": voice,
            "spiral_score": spiral,
            "gait_score": gait,
            "tapping_score": tapping,
            "final_risk": risk_score / 100,
            "severity": severity
        }).execute()
        
    return {
        "risk_score": risk_score,
        "severity": severity,
        "explanation": explanation
    }

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
