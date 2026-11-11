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
    "image-generation", "image_gen", "video-generation", "active-speaker",
    "bevformer", "cosmos", "streampetr", "synthetic-video", "studiovoice",
    "kumo-relational", "kumo-tabular", "ising-calibration", "nvidia/bnr",
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

def _normalize_model_id(model_id: str) -> str:
    # NGC slugs sometimes use hyphens where the OpenAI-compatible API uses
    # version dots (e.g. glm-5-3-flash vs glm-5.3-flash).
    return model_id.lower().replace("glm-5-3", "glm-5.3")


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
    # Walk all NGC catalog pages; the live NVIDIA /models list remains the
    # source of truth for models currently exposed to this API key.
    page_size = 200
    max_pages = 20
    catalog_ids: set[str] = set()

    async def fetch_catalog_page(page: int) -> tuple[set[str], int]:
        query = {
            "query": "*",
            "filters": [{"field": "label", "value": "nim_type_preview"}],
            "orderBy": [{"field": "score", "value": "DESC"}],
            "page": page,
            "pageSize": page_size,
            "scoredSize": page_size,
        }
        catalog_url = (
            f"{NVIDIA_CATALOG_URL}?q={quote(json.dumps(query), safe='')}"
            "&group-labels-by-labelset=true"
        )
        response = await app.state.http.get(
            catalog_url, headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        payload = response.json()
        resource_count = sum(
            len(group.get("resources", []))
            for group in payload.get("results", [])
            if isinstance(group, dict) and isinstance(group.get("resources", []), list)
        )
        return _catalog_model_ids(payload), resource_count

    models_task = app.state.http.get(
        f"{settings.nvidia_base_url.rstrip('/')}/models",
        headers=upstream_headers("nvidia", settings.nvidia_api_key),
    )
    models_response, first_page = await asyncio.gather(
        models_task, fetch_catalog_page(0)
    )
    models_response.raise_for_status()

    page_ids, resource_count = first_page
    catalog_ids.update(page_ids)

    # Continue until a short/empty page, a repeated page, or a safety cap.
    for page in range(1, max_pages):
        if resource_count < page_size:
            break
        page_ids, resource_count = await fetch_catalog_page(page)
        new_ids = page_ids - catalog_ids
        if not new_ids:
            break
        catalog_ids.update(new_ids)

    if not catalog_ids:
        raise ValueError("NVIDIA free-endpoint catalog returned no model IDs.")

    upstream = models_response.json()
    available = []
    for model in upstream.get("data", []):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        model_id = model["id"]
        lowered = model_id.lower()
        if not any(_normalize_model_id(item) == _normalize_model_id(lowered) for item in catalog_ids):
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

@app.get("/v1/models/diagnostics")
async def model_diagnostics(authorization: str | None = Header(default=None)):
    authorize(authorization)
    if not settings.nvidia_api_key:
        raise HTTPException(503, "NVIDIA_API_KEY is not configured on the bridge.")

    # Compare the authenticated upstream list against the NGC free-endpoint
    # catalog without changing the public model list or request routing.
    page_size = 200
    catalog_ids: set[str] = set()
    pages = 0
    for page in range(20):
        query = {
            "query": "*",
            "filters": [{"field": "label", "value": "nim_type_preview"}],
            "orderBy": [{"field": "score", "value": "DESC"}],
            "page": page,
            "pageSize": page_size,
            "scoredSize": page_size,
        }
        url = (
            f"{NVIDIA_CATALOG_URL}?q={quote(json.dumps(query), safe='')}"
            "&group-labels-by-labelset=true"
        )
        try:
            response = await app.state.http.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, f"Catalog diagnostics failed: {type(exc).__name__}")
        page_ids = _catalog_model_ids(payload)
        resource_count = sum(
            len(group.get("resources", []))
            for group in payload.get("results", [])
            if isinstance(group, dict) and isinstance(group.get("resources", []), list)
        )
        pages += 1
        new_ids = page_ids - catalog_ids
        catalog_ids.update(page_ids)
        if resource_count < page_size or not new_ids:
            break

    try:
        response = await app.state.http.get(
            f"{settings.nvidia_base_url.rstrip('/')}/models",
            headers=upstream_headers("nvidia", settings.nvidia_api_key),
        )
        response.raise_for_status()
        upstream = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, f"Upstream diagnostics failed: {type(exc).__name__}")

    live_ids = {
        model["id"].lower()
        for model in upstream.get("data", [])
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    }
    chat_catalog = {
        model for model in catalog_ids
        if not any(token in model for token in _non_chat_tokens)
    }
    return {
        "catalog_pages_scanned": pages,
        "catalog_free_endpoint_count": len(catalog_ids),
        "catalog_chat_candidate_count": len(chat_catalog),
        "nvidia_live_model_count": len(live_ids),
        "intersection_count": len({_normalize_model_id(x) for x in chat_catalog} & {_normalize_model_id(x) for x in live_ids}),
        "catalog_chat_missing_from_live": sorted(x for x in chat_catalog if _normalize_model_id(x) not in {_normalize_model_id(y) for y in live_ids}),
        "live_missing_from_catalog": sorted(x for x in live_ids if _normalize_model_id(x) not in {_normalize_model_id(y) for y in catalog_ids}),
    }


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
