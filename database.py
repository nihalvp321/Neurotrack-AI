import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")

supabase = None
supabase_admin = None

try:
    if url and key:
        supabase = create_client(url, key)
        print("Successfully initialized Supabase client")
    else:
        print("CRITICAL: SUPABASE_URL or SUPABASE_KEY is missing from environment")

    if url and service_key:
        supabase_admin = create_client(url, service_key)
        print("Successfully initialized Supabase admin client")
    else:
        supabase_admin = supabase
        print("WARNING: SUPABASE_SERVICE_KEY is missing, falling back to standard client")
except Exception as e:
    print(f"FATAL ERROR during Supabase initialization: {e}")


def get_supabase():
    return supabase

def get_supabase_admin():
    return supabase_admin
