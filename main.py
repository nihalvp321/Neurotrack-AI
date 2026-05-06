from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import auth
import os
import uuid
import joblib
import numpy as np
from database import get_supabase, get_supabase_admin
from voice_analysis import extract_voice_features
from spiral_analysis import extract_spiral_features
from tremor_analysis import extract_tremor_features
from gait_analysis import extract_gait_features

app = FastAPI(title="Neurotrack AI API")

# ── LAZY MODEL LOADING ────────────────────────────────────
# We load models only when needed to save RAM on Render (Free tier limit: 512MB)

voice_model = None
def get_voice_model():
    global voice_model
    if voice_model is None:
        VOICE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "neurotrack-ml", "models", "voice model", "voice_model.pkl")
        try:
            voice_model = joblib.load(VOICE_MODEL_PATH)
            print(f"--- Voice Model Loaded ---")
        except Exception as e:
            print(f"--- Failed to load Voice Model: {e} ---")
    return voice_model

spiral_scaler = None
spiral_pca = None
spiral_svm = None
def get_spiral_model():
    global spiral_scaler, spiral_pca, spiral_svm
    if any(m is None for m in [spiral_scaler, spiral_pca, spiral_svm]):
        SPIRAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "neurotrack-ml", "models", "spiral model")
        try:
            spiral_scaler = joblib.load(os.path.join(SPIRAL_MODEL_DIR, "spiral_scaler.pkl"))
            spiral_pca = joblib.load(os.path.join(SPIRAL_MODEL_DIR, "spiral_pca.pkl"))
            spiral_svm = joblib.load(os.path.join(SPIRAL_MODEL_DIR, "spiral_svm_model.pkl"))
            print(f"--- Spiral Model Loaded ---")
        except Exception as e:
            print(f"--- Failed to load Spiral Model: {e} ---")
    return spiral_scaler, spiral_pca, spiral_svm

tremor_model = None
tremor_feature_names = None
def get_tremor_model():
    global tremor_model, tremor_feature_names
    if tremor_model is None:
        TREMOR_MODEL_DIR = os.path.join(os.path.dirname(__file__), "neurotrack-ml", "models", "tremor_model")
        try:
            tremor_model = joblib.load(os.path.join(TREMOR_MODEL_DIR, "tremor_classifier.pkl"))
            tremor_feature_names = joblib.load(os.path.join(TREMOR_MODEL_DIR, "tremor_feature_names.pkl"))
            print(f"--- Tremor Model Loaded ---")
        except Exception as e:
            print(f"--- Failed to load Tremor Model: {e} ---")
    return tremor_model, tremor_feature_names

gait_model = None
gait_feature_names = None
def get_gait_model():
    global gait_model, gait_feature_names
    if gait_model is None:
        GAIT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "neurotrack-ml", "models", "gait_model")
        try:
            gait_model = joblib.load(os.path.join(GAIT_MODEL_DIR, "gait_model.pkl"))
            gait_feature_names = joblib.load(os.path.join(GAIT_MODEL_DIR, "gait_feature_names.pkl"))
            print(f"--- Gait Model Loaded ---")
        except Exception as e:
            print(f"--- Failed to load Gait Model: {e} ---")
    return gait_model, gait_feature_names

ensemble_model = None
def get_ensemble_model():
    global ensemble_model
    if ensemble_model is None:
        ENSEMBLE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "neurotrack-ml", "models", "ensemble", "ensemble_model.pkl")
        try:
            ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
            print(f"--- Ensemble Meta-Model Loaded ---")
        except Exception as e:
            print(f"--- Failed to load Ensemble Model: {e} ---")
    return ensemble_model

VOICE_FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# ── SCORING & CALIBRATION HELPERS ─────────────────────────
def calibrate_score(prob, power=2.1, threshold=0.92):
    """
    Refined calibration (power 2.1) to improve sensitivity.
    Previous power (3.2) was too conservative, causing 'bad' tests
    to appear as low risk. 2.1 maintains a healthy baseline while
    allowing moderate abnormalities to surface more clearly.
    """
    if prob >= threshold:
        return float(prob)
    # 0.7 raw -> (0.7/0.92)^2.1 * 0.92 = 0.52 (previously ~0.38)
    # 0.8 raw -> (0.8/0.92)^2.1 * 0.92 = 0.69 (previously ~0.59)
    scaled = (prob / threshold) ** power * threshold
    return float(scaled)

def get_severity_assessment(risk_percent):
    if risk_percent < 25:
        return {'stage': 'Low Risk', 'hoehn_yahr': 'N/A', 'advice': 'No significant Parkinson\'s indicators detected.'}
    elif risk_percent < 45:
        return {'stage': 'Mild Risk', 'hoehn_yahr': 'Stage 1 (possible)', 'advice': 'Some mild indicators present. Consider a full evaluation.'}
    elif risk_percent < 65:
        return {'stage': 'Moderate Risk', 'hoehn_yahr': 'Stage 1-2 (possible)', 'advice': 'Moderate indicators detected. Neurologist consultation is recommended.'}
    elif risk_percent < 80:
        return {'stage': 'High Risk', 'hoehn_yahr': 'Stage 2-3 (possible)', 'advice': 'Strong Parkinson\'s indicators. Please see a neurologist soon.'}
    else:
        return {'stage': 'Very High Risk', 'hoehn_yahr': 'Stage 3+ (possible)', 'advice': 'Very strong indicators. Urgent consultation is strongly recommended.'}

# Configure CORS for Flutter (allow all for development)
# Logging Middleware to debug connection issues
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"--- Incoming Request: {request.method} {request.url} ---")
    response = await call_next(request)
    print(f"--- Response Status: {response.status_code} ---")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Neurotrack AI API is running"}

app.include_router(auth.router)

@app.post("/predict/voice")
async def predict_voice(email: str, file: UploadFile = File(...)):
    model = get_voice_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Voice model not loaded on server")

    # 1. Save temp file to analyze
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 2. Extract Features (22 clinical metrics)
        features = extract_voice_features(temp_path)
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract features. Is the audio too short or silent?")

        # 3. Predict
        import pandas as pd
        features_df = pd.DataFrame([features], columns=VOICE_FEATURE_NAMES)
        
        # Get probability (probability=True was set in SVC)
        prob = model.predict_proba(features_df)[0][1]
        
        # --- NOISE & QUALITY PENALTY ---
        # HNR (Harmonic-to-Noise Ratio) is at index 15. 
        # Healthy is usually > 20. PD is often < 15. 
        # If background noise or heavy breathing is present, HNR drops.
        hnr_val = features[15]
        if hnr_val < 15.0:
            # Gradually increase probability if signal quality is poor/abnormal
            penalty = (15.0 - hnr_val) / 30.0 # Max penalty ~0.5
            prob = min(0.95, prob + penalty)
            print(f"--- Voice Quality Penalty Applied (HNR: {hnr_val:.2f}, Penalty: {penalty:.2f}) ---")

        prediction = 1 if prob >= 0.5 else 0

        # 4. Upload to Supabase Storage (Non-blocking for history)
        audio_url = None
        try:
            supabase = get_supabase_admin()
            file_path = f"{email}/voice_tests/{file.filename}"
            with open(temp_path, "rb") as f:
                supabase.storage.from_("voice-tests").upload(
                    path=file_path,
                    file=f,
                    file_options={"content-type": "audio/wav", "upsert": "true"}
                )
            audio_url = supabase.storage.from_("voice-tests").get_public_url(file_path)
        except Exception as se:
            print(f"--- [Storage Upload Failed (Voice)] {se} ---")

        return {
            "score": calibrate_score(prob),
            "prediction": prediction,
            "label": "Parkinson's Detected" if prediction == 1 else "Healthy",
            "features": features,
            "audio_url": audio_url
        }

    except Exception as e:
        import traceback
        print(f"--- [Voice Prediction ERROR] ---")
        print(f"Error Type: {type(e)}")
        print(f"Error Detail: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict/spiral")
async def predict_spiral(email: str, file: UploadFile = File(...)):
    scaler, pca, svm = get_spiral_model()
    if any(m is None for m in [scaler, pca, svm]):
        raise HTTPException(status_code=500, detail="Spiral model components (scaler/pca/svm) not loaded on server")

    # 1. Save temp file to analyze
    temp_path = f"temp_spiral_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 2. Extract Features (29 clinical metrics)
        features = extract_spiral_features(temp_path)
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract features from spiral image.")

        # 3. Preprocess and Predict
        # Pipeline expects a 2D array [1, 29]
        features_2d = features.reshape(1, -1)
        
        # Apply the exact same transformation as training
        scaled_features = scaler.transform(features_2d)
        pca_features = pca.transform(scaled_features)
        
        # Get probability
        prob = svm.predict_proba(pca_features)[0][1]
        
        # --- DIGITAL JERK & CONTINUITY PENALTY ---
        # Features: Index 14 = num_contours, Index 11 = solidity
        # A healthy spiral should be 1-3 continuous contours.
        # Shaky hands create many broken contours.
        num_c = features[14]
        solidity = features[11]
        
        if num_c > 12:
            penalty = min(0.3, (num_c - 12) * 0.03)
            prob = min(0.98, prob + penalty)
            print(f"--- Spiral Jerk Penalty Applied (Contours: {num_c}, Penalty: {penalty:.2f}) ---")
        
        if solidity < 0.65:
            # Solidity threshold reduced to 0.65 to be less aggressive for clean drawings
            s_penalty = (0.65 - solidity) * 0.4
            prob = min(0.98, prob + s_penalty)
            print(f"--- Spiral Solidity Penalty Applied (Solidity: {solidity:.2f}, Penalty: {s_penalty:.2f}) ---")

        prediction = 1 if prob >= 0.5 else 0

        # 4. Upload to Supabase Storage (Non-blocking for history)
        image_url = None
        try:
            supabase = get_supabase_admin()
            file_path = f"{email}/spiral_tests/{file.filename}"
            with open(temp_path, "rb") as f:
                supabase.storage.from_("voice-tests").upload(
                    path=file_path,
                    file=f,
                    file_options={"content-type": "image/png", "upsert": "true"}
                )
            image_url = supabase.storage.from_("voice-tests").get_public_url(file_path)
        except Exception as se:
            print(f"--- [Storage Upload Failed (Spiral)] {se} ---")

        return {
            "score": calibrate_score(prob),
            "prediction": prediction,
            "label": "Parkinson's Detected" if prediction == 1 else "Healthy",
            "features": features.tolist(),
            "image_url": image_url
        }

    except Exception as e:
        print(f"[Spiral Prediction Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


class TremorData(BaseModel):
    email: str
    x_values: List[float]
    y_values: List[float]
    z_values: List[float]
    timestamps: List[int] = []

@app.post("/predict/tremor")
async def predict_tremor(data: TremorData):
    model, feature_names = get_tremor_model()
    if model is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Tremor model not loaded on server")
    
    try:
        # Calculate dynamic sampling frequency
        fs = 50 # Default
        if len(data.timestamps) > 10:
            duration = (data.timestamps[-1] - data.timestamps[0]) / 1000.0
            if duration > 0:
                fs = len(data.timestamps) / duration
                print(f"--- Detected Tremor FS: {fs:.2f} Hz ---")
        
        # 1. Extract Tremor Features
        # Convert m/s^2 to G-units (assuming model was trained on Gs)
        x_g = np.array(data.x_values) / 9.8
        y_g = np.array(data.y_values) / 9.8
        z_g = np.array(data.z_values) / 9.8
        
        features = extract_tremor_features(
            x_g,
            y_g,
            z_g,
            fs=fs
        )
        
        if features is None:
            raise HTTPException(status_code=400, detail="Insufficient sensor data (need at least 2.5 seconds)")

        # 2. Predict
        # Convert features list to a DataFrame with correct names (crucial for Pipeline)
        import pandas as pd
        features_df = pd.DataFrame([features], columns=tremor_feature_names)
        
        # Use .values to avoid UserWarning about feature names if the scaler was fitted on arrays
        prob = model.predict_proba(features_df.values)[0][1]
        prediction = int(model.predict(features_df.values)[0])

        return {
            "score": calibrate_score(prob),
            "prediction": prediction,
            "label": "Parkinson's Detected" if prediction == 1 else "Healthy",
            "dominant_freq": float(features[0]), # The first feature is dominant_freq
            "features": features
        }

    except Exception as e:
        print(f"[Tremor Prediction Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/gait")
async def predict_gait(email: str, file: UploadFile = File(...)):
    model, feature_names = get_gait_model()
    if model is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Gait model not loaded on server")

    # 1. Save temp file
    temp_path = f"temp_gait_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 2. Extract Features
        features = extract_gait_features(temp_path)
        if features is None:
            raise HTTPException(status_code=400, detail="Insufficient or invalid gait data. Ensure the file has 19 columns and sufficient length.")

        # 3. Predict
        import pandas as pd
        features_df = pd.DataFrame([features], columns=gait_feature_names)
        
        # Use .values to avoid UserWarning
        prob = model.predict_proba(features_df.values)[0][1]
        prediction = int(model.predict(features_df.values)[0])

        return {
            "score": calibrate_score(prob),
            "prediction": prediction,
            "label": "Parkinson's Detected" if prediction == 1 else "Healthy",
            "features": features
        }

    except Exception as e:
        print(f"[Gait Prediction Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)




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

@app.get("/sessions/history")
async def get_session_history(email: str, limit: int = 10, offset: int = 0):
    supabase = get_supabase()
    user_res = supabase.table("users").select("id").eq("email", email).execute()
    if not user_res.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user_res.data[0]["id"]
    
    # Apply pagination: limit and range(offset, offset + limit - 1)
    session_res = supabase.table("test_sessions")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()
        
    return session_res.data

@app.post("/predict")
async def predict(data: dict):
    model = get_ensemble_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Ensemble meta-model not loaded on server")

    # 1. Collect scores from the 4 tests
    voice = data.get('voice_score', 0)
    spiral = data.get('spiral_score', 0)
    tremor = data.get('tremor_score', 0)
    gait = data.get('gait_score', 0)
    
    # 2. Feed into Ensemble Meta-Classifier
    # Order must match training: [voice, spiral, tremor, gait]
    scores_input = np.array([[voice, spiral, tremor, gait]])
    
    try:
        # Get decision function (log-odds)
        decision = model.decision_function(scores_input)[0]
        # Offset removed to prevent moderate scores from being pushed to 0%
        risk_prob = 1 / (1 + np.exp(-decision))
        
        # --- SMART SAFEGUARD ---
        # Ensure final risk is at least a portion of the mean individual scores
        max_score = np.max(scores_input)
        mean_score = np.mean(scores_input)
        
        if max_score > 0.40:
            # If any test is notably high (over 40%), the overall risk
            # should lean more heavily into that specific concern.
            risk_prob = max(risk_prob, max_score * 0.8, mean_score * 0.9)
        elif mean_score > 0.12:
            # For low-moderate profiles, anchor it higher to the mean
            risk_prob = max(risk_prob, mean_score * 0.7)        
        print(f"--- Ensemble Input (Raw): {scores_input} ---")
        print(f"--- Ensemble Adjusted Risk: {risk_prob:.2%} ---")
        
    except Exception as e:
        print(f"Ensemble prediction error: {e}")
        risk_prob = np.mean([voice, spiral, tremor, gait])
        
    risk_percent = float(risk_prob * 100)
    
    # 3. Get Clinical Mapping
    assessment = get_severity_assessment(risk_percent)
    
    explanation = f"Your AI assessment shows a {assessment['stage'].lower()} profile ({risk_percent:.1f}% risk). "
    if assessment['hoehn_yahr'] != 'N/A':
        explanation += f"This potentially correlates with {assessment['hoehn_yahr']}. "
    explanation += f"\n\nRecommendation: {assessment['advice']}"
    explanation += "\n\nNote: This is an AI screening tool, not a formal medical diagnosis."
    
    # Store session in DB - Wrapped in try-except to prevent 500 if DB is unstable
    try:
        supabase = get_supabase_admin()
        user_res = supabase.table("users").select("id").eq("email", data.get('user_id')).execute()
        if user_res.data:
            user_id = user_res.data[0]["id"]
            session_id = str(uuid.uuid4())
            print(f"Found user ID: {user_id}. Attempting insert...")
            supabase.table("test_sessions").insert({
                "id": session_id,
                "user_id": user_id,
                "tremor_score": tremor,
                "voice_score": voice,
                "spiral_score": spiral,
                "gait_score": gait,
                "tapping_score": 0,
                "final_risk": risk_prob,
                "severity_stage": assessment['stage'],
                "explanation": explanation
            }).execute()
            print("--- DATABASE SYNC SUCCESSFUL ---")
    except Exception as e:
        print(f"--- DATABASE SYNC FAILED (NON-CRITICAL): {str(e)} ---")
            
    return {
        "risk_score": risk_percent,
        "severity": assessment['stage'],
        "explanation": explanation,
        "hoehn_yahr": assessment['hoehn_yahr'],
        "individual_scores": {
            "voice": voice,
            "spiral": spiral,
            "tremor": tremor,
            "gait": gait
        },
        "details": {
            "voice_features": data.get('voice_features'),
            "spiral_features": data.get('spiral_features'),
            "tremor_features": data.get('tremor_features'),
            "gait_features": data.get('gait_features')
        }
    }

# ── CHAT HISTORY ENDPOINTS ──────────────────────────────



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
