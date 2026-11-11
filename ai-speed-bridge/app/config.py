import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    bridge_api_key: str = os.getenv("BRIDGE_API_KEY", "")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    default_provider: str = os.getenv("DEFAULT_PROVIDER", "nvidia").lower()
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "300"))
    max_connections: int = int(os.getenv("MAX_CONNECTIONS", "100"))
    max_keepalive_connections: int = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "20"))

settings = Settings()
