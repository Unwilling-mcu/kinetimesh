from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "KinetiMesh API"
    VERSION: str = "3.0.0"
    DEBUG: bool = False
    DATABASE_URL: str = "postgresql+asyncpg://kinetimesh:kinetimesh@localhost:5432/kinetimesh"
    REDIS_URL: str = "redis://localhost:6379"
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    SECRET_KEY: str = "dev-secret-change-in-production"
    CORS_ORIGINS: List[str] = ["http://localhost:3000","http://localhost:5173"]
    FL_MU: float = 0.01
    FL_LOCAL_EPOCHS: int = 5
    RL_DISPATCH_INTERVAL: int = 60
    QAOA_ENABLED: bool = False
    QAOA_DEPTH: int = 3
    class Config:
        env_file = ".env"

settings = Settings()
