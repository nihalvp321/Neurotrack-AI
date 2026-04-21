from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import auth

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

@app.get("/")
async def root():
    return {"message": "Welcome to Neurotrack AI API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
