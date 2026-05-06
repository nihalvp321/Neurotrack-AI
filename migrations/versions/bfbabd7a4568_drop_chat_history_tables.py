"""drop_chat_history_tables

Revision ID: bfbabd7a4568
Revises: 2ee7b358783f
Create Date: 2026-05-06 20:14:10.919740

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bfbabd7a4568'
down_revision: Union[str, Sequence[str], None] = '2ee7b358783f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_table('chat_messages')
    op.drop_table('chat_sessions')


def downgrade() -> None:
    """Downgrade schema."""
    pass
