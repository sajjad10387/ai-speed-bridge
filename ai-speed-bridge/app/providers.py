from typing import Any
from .config import settings

class ProviderError(Exception):
    pass

def provider_config(provider: str) -> tuple[str, str]:
    provider = provider.lower()
    if provider == "nvidia":
        if not settings.nvidia_api_key:
            raise ProviderError("NVIDIA_API_KEY is not configured on the bridge.")
        return settings.nvidia_base_url.rstrip("/"), settings.nvidia_api_key
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not configured on the bridge.")
        return settings.openrouter_base_url.rstrip("/"), settings.openrouter_api_key
    raise ProviderError("Unsupported provider. Use nvidia or openrouter.")

def upstream_headers(provider: str, api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if provider.lower() == "openrouter":
        headers["HTTP-Referer"] = "https://render.com"
        headers["X-Title"] = "Personal AI Speed Bridge"
    return headers
