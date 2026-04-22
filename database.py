import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")

if not url or not key:
    print("WARNING: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")

supabase: Client = create_client(url, key) if url and key else None
# Admin client uses service_role key — bypasses RLS for backend operations like storage uploads
supabase_admin: Client = create_client(url, service_key) if url and service_key else supabase

def get_supabase():
    return supabase

def get_supabase_admin():
    return supabase_admin
