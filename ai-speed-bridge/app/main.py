import asyncio
import hmac
import json
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import settings
from .providers import provider_config, upstream_headers, ProviderError

NVIDIA_CATALOG_URL = "https://api.ngc.nvidia.com/v2/search/catalog/resources/ENDPOINT"
NVIDIA_NGC_ORG = "qc69jvmznzxy"
MODEL_CACHE_SECONDS = 600
_non_chat_tokens = (
    "embed", "rerank", "safety", "guard", "translate", "transcribe",
    "speech", "asr", "tts", "ocr", "parse", "pose", "protein",
    "image-generation", "image_gen", "video-generation",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(
        max_connections=settings.max_connections,
        max_keepalive_connections=settings.max_keepalive_connections,
        keepalive_expiry=30.0,
    )
    timeout = httpx.Timeout(settings.request_timeout, connect=15.0)
    app.state.http = httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        http2=True,
        follow_redirects=False,
    )
    app.state.model_cache = {"expires": 0.0, "data": None}
    app.state.model_lock = asyncio.Lock()
    yield
    await app.state.http.aclose()

app = FastAPI(title="Personal AI Speed Bridge", version="1.0.0", lifespan=lifespan)

def authorize(authorization: str | None):
    if not settings.bridge_api_key:
        raise HTTPException(503, "BRIDGE_API_KEY is not configured.")
    expected = f"Bearer {settings.bridge_api_key}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(401, "Invalid bridge API key.")

def _catalog_model_ids(payload: dict) -> set[str]:
    ids = set()
    for group in payload.get("results", []):
        for resource in group.get("resources", []) if isinstance(group, dict) else []:
            if not isinstance(resource, dict):
                continue
            name = resource.get("name")
            publisher = resource.get("publisher")
            if not publisher:
                for label in resource.get("labels", []):
                    if isinstance(label, dict) and label.get("key") == "publisher":
                        values = label.get("values", []) + label.get("unresolvedValues", [])
                        if values:
                            publisher = values[0]
                            break
            if isinstance(name, str) and isinstance(publisher, str):
                ids.add(f"{publisher}/{name}".lower())
    return ids

async def _fetch_free_models():
    # NVIDIA's NGC catalog uses nim_type_preview for hosted preview endpoints.
    # This includes free endpoints and free-plus-downloadable endpoints.
    # Downloadable-only and partner-only entries are excluded.
    query = {
        "query": f'orgName:"{NVIDIA_NGC_ORG}"',
        "filters": [{"field": "label", "value": "nim_type_preview"}],
        "orderBy": [{"field": "score", "value": "DESC"}],
        "page": 0,
        "pageSize": 200,
        "scoredSize": 200,
    }
    catalog_url = (
        f"{NVIDIA_CATALOG_URL}?q={quote(json.dumps(query), safe='')}"
        "&group-labels-by-labelset=true"
    )
    catalog_response, models_response = await asyncio.gather(
        app.state.http.get(catalog_url, headers={"Accept": "application/json"}),
        app.state.http.get(
            f"{settings.nvidia_base_url.rstrip('/')}/models",
            headers=upstream_headers("nvidia", settings.nvidia_api_key),
        ),
    )
    catalog_response.raise_for_status()
    models_response.raise_for_status()
    catalog = catalog_response.json()
    upstream = models_response.json()
    free_ids = _catalog_model_ids(catalog)
    if not free_ids:
        raise ValueError("NVIDIA free-endpoint catalog returned no model IDs.")

    available = []
    for model in upstream.get("data", []):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        model_id = model["id"]
        lowered = model_id.lower()
        if lowered not in free_ids:
            continue
        if any(token in lowered for token in _non_chat_tokens):
            continue
        available.append(model)

    return sorted(available, key=lambda item: item["id"].lower())

@app.get("/")
async def root():
    return {"service": "Personal AI Speed Bridge", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    authorize(authorization)
    if not settings.nvidia_api_key:
        raise HTTPException(503, "NVIDIA_API_KEY is not configured on the bridge.")

    cache = app.state.model_cache
    if cache["data"] is not None and cache["expires"] > time.time():
        return {"object": "list", "data": cache["data"]}

    async with app.state.model_lock:
        cache = app.state.model_cache
        if cache["data"] is not None and cache["expires"] > time.time():
            return {"object": "list", "data": cache["data"]}
        try:
            data = await _fetch_free_models()
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            # Never fall back to the unfiltered NVIDIA list: that could expose paid models.
            raise HTTPException(502, f"Could not refresh NVIDIA free model catalog: {type(exc).__name__}")
        app.state.model_cache = {"expires": time.time() + MODEL_CACHE_SECONDS, "data": data}
        return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    authorize(authorization)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be valid JSON.")

    model = payload.get("model", "")
    if not isinstance(model, str) or not model:
        raise HTTPException(400, "A model is required.")
    if model.startswith("nvidia/"):
        provider = "nvidia"
        upstream_model = model
    elif model.startswith("openrouter/"):
        provider = "openrouter"
        upstream_model = model[len("openrouter/"):]
    else:
        provider = settings.default_provider
        upstream_model = model

    try:
        base_url, api_key = provider_config(provider)
    except ProviderError as exc:
        raise HTTPException(503, str(exc))

    payload["model"] = upstream_model
    headers = upstream_headers(provider, api_key)
    url = f"{base_url}/chat/completions"

    try:
        upstream_request = app.state.http.build_request("POST", url, json=payload, headers=headers)
        upstream_response = await app.state.http.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        raise HTTPException(504, "Provider connection timed out.")
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Provider connection failed: {type(exc).__name__}")

    response_headers = {}
    for key in ("cache-control", "x-request-id", "openrouter-processing-ms"):
        if key in upstream_response.headers:
            response_headers[key] = upstream_response.headers[key]

    async def body_iterator():
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        body_iterator(),
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type", "text/event-stream"),
        headers=response_headers,
    )
