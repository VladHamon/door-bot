import os
import asyncpg
from datetime import datetime
from urllib.parse import urlparse


class Database:
    def __init__(self):
        self.pool = None
        self.db_url = os.environ.get("DATABASE_URL")

    async def connect(self):
        """Создает пул соединений к БД."""
        if not self.db_url:
            print("⚠️ DATABASE_URL not found, analytics disabled.")
            return
        
        try:
            self.pool = await asyncpg.create_pool(self.db_url)
            print("✅ Connected to PostgreSQL")
            await self.create_tables()
        except Exception as e:
            print(f"❌ DB Connection Error: {e}")

    async def close(self):
        """Закрывает соединения."""
        if self.pool:
            await self.pool.close()

    async def create_tables(self):
        """Создает таблицу для аналитики, если её нет."""
        query = """
        CREATE TABLE IF NOT EXISTS user_events (
            id SERIAL PRIMARY KEY,
            user_id BIGINT,
            username TEXT,
            event_type TEXT,      -- 'command', 'click', 'generation'
            event_name TEXT,      -- '/start', 'color:red', 'generate_door'
            details TEXT,         -- Доп. инфо (какой промпт, какой ID двери)
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query)

    async def log(self, user_id: int, username: str, event_type: str, event_name: str, details: str = None):
        """Записывает событие в базу."""
        if not self.pool:
            return

        query = """
        INSERT INTO user_events (user_id, username, event_type, event_name, details)
        VALUES ($1, $2, $3, $4, $5)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, user_id, username, event_type, event_name, details)
        except Exception as e:
            print(f"⚠️ Failed to log event: {e}")

# Создаем глобальный экземпляр
db = Database()
