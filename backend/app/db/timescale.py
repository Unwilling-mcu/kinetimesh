import asyncpg, logging
from app.core.config import settings
log = logging.getLogger("kinetimesh.db")

_pool = None

async def init_db():
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL.replace("postgresql+asyncpg://","postgresql://"),
            min_size=2, max_size=10, command_timeout=60
        )
        log.info("Database pool initialized")
    except Exception as e:
        log.warning(f"DB not available (simulation mode): {e}")

async def get_pool():
    return _pool

async def fetch(query: str, *args):
    if _pool is None: return []
    async with _pool.acquire() as conn:
        return await conn.fetch(query, *args)

async def execute(query: str, *args):
    if _pool is None: return None
    async with _pool.acquire() as conn:
        return await conn.execute(query, *args)
