"""add voice tests bucket

Revision ID: b8d8f0f0f0f0
Revises: 5f17cbc3b506
Create Date: 2026-04-24 08:18:00.000000

"""
from alembic import op
import sqlalchemy as sa
from database import get_supabase_admin

# revision identifiers, used by Alembic.
revision = 'b8d8f0f0f0f0'
down_revision = '5f17cbc3b506'
branch_labels = None
depends_on = None

def upgrade():
    # Use the Supabase admin client to create the storage bucket
    admin = get_supabase_admin()
    try:
        # Create a public bucket for voice recordings
        admin.storage.create_bucket('voice-tests', options={'public': True})
        print('Supabase Storage: Bucket "voice-tests" created.')
    except Exception as e:
        # If bucket already exists or other error, just print it
        print(f'Supabase Storage: {e}')

def downgrade():
    # In a real scenario, we might want to delete the bucket, 
    # but that's dangerous so we'll just skip it for now.
    pass
