import hmac
import json
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .config import settings
from .providers import provider_config, upstream_headers, ProviderError

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
    yield
    await app.state.http.aclose()

app = FastAPI(title="Personal AI Speed Bridge", version="1.0.0", lifespan=lifespan)

def authorize(authorization: str | None):
    if not settings.bridge_api_key:
        raise HTTPException(503, "BRIDGE_API_KEY is not configured.")
    expected = f"Bearer {settings.bridge_api_key}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(401, "Invalid bridge API key.")

@app.get("/")
async def root():
    return {"service": "Personal AI Speed Bridge", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    authorize(authorization)
    # Copilot/custom endpoint model discovery can use this static list.
    return {
        "object": "list",
        "data": [
            {"id": "nvidia/llama-3.3-70b-instruct", "object": "model", "created": 0, "owned_by": "nvidia"},
            {"id": "openrouter/auto", "object": "model", "created": 0, "owned_by": "openrouter"},
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    authorize(authorization)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be valid JSON.")

    # Select provider with a model prefix: nvidia/<model> or openrouter/<model>.
    model = payload.get("model", "")
    if not isinstance(model, str) or not model:
        raise HTTPException(400, "A model is required.")
    if model.startswith("nvidia/"):
        provider = "nvidia"
        # NVIDIA NIM expects the full model ID, including the "nvidia/" namespace.
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
    # Do not force stream=true: preserve the client's requested response mode.
    headers = upstream_headers(provider, api_key)
    url = f"{base_url}/chat/completions"

    try:
        upstream_request = app.state.http.build_request(
            "POST", url, json=payload, headers=headers
        )
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
