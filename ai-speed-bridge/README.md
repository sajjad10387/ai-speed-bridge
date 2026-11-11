# Personal AI Speed Bridge

A lightweight OpenAI-compatible streaming proxy intended for a personal VS Code setup.

## What it does

- Accepts `POST /v1/chat/completions`.
- Routes `nvidia/<model-id>` to NVIDIA NIM.
- Routes `openrouter/<model-id>` to OpenRouter.
- Streams upstream bytes back immediately (no response buffering).
- Reuses async HTTP connections and enables HTTP/2.
- Protects the bridge with a separate `BRIDGE_API_KEY`.
- Exposes `/health` for uptime monitoring.
- Dynamically discovers NVIDIA models marked `Free Endpoint`, including models also marked `Downloadable`; excludes downloadable-only and partner-only models.
- Intersects NVIDIA's free-endpoint catalog with the live `/v1/models` API and caches the result for 10 minutes.

## Important compatibility note

This bridge is a Chat Completions-compatible proxy. It does **not** guarantee compatibility with every GitHub Copilot custom-provider mode or every Agent Mode tool-call flow. VS Code's provider support and Copilot model integration can vary by version and provider API. Test ordinary chat, streaming, tool calls, images, and long context before relying on it.

It does not make NVIDIA/OpenRouter model inference faster. It can reduce avoidable proxy overhead and provide a stable nearby connection only if the Render region/network route is favorable. Render Free can cold-start, restart, throttle CPU, and has monthly instance-hour limits; an uptime ping may prevent idle sleep in some cases but is not a guarantee of uninterrupted availability.

## Deploy to Render

1. Create a private GitHub repository and push this project.
2. In Render, choose **New > Blueprint**, connect the repository, and deploy `render.yaml`.
3. In Render service Environment, set:
   - `NVIDIA_API_KEY`
   - `OPENROUTER_API_KEY` (optional)
   - `BRIDGE_API_KEY` (Render generates one from the blueprint; copy it securely)
4. Deploy and wait for `/health` to return `{"status":"ok",...}`.
5. Configure UptimeRobot to monitor `https://YOUR-SERVICE.onrender.com/health` every 5 minutes if permitted by Render's current free-tier terms. Do not assume this guarantees 24/7 uptime.


## Model discovery

`GET /v1/models` requires the bridge bearer key. It dynamically returns NVIDIA models that have the NGC `nim_type_preview` label (Free Endpoint), intersected with NVIDIA's live model API. This includes **Free Endpoint** and **Free Endpoint + Downloadable**, but excludes **Downloadable only** and partner-only entries. Results are cached for 10 minutes. If NVIDIA's catalog cannot be refreshed, the bridge returns an error rather than accidentally exposing unfiltered models.

## Connect VS Code Copilot Chat (Custom Endpoint)

In VS Code, open **Chat: Manage Language Models** → **Add Models** → **Custom Endpoint**. Choose **Chat Completions**. The custom endpoint UI in some VS Code versions requires model entries to be declared manually; it may not automatically populate its picker from `/v1/models`. Other OpenAI-compatible clients that support model discovery can fetch the dynamic list automatically.

Add an entry like this to `chatLanguageModels.json` (use the bridge URL and keep the bridge key in VS Code's input secret prompt):

```json
[
  {
    "name": "My Render AI Bridge",
    "vendor": "customendpoint",
    "apiKey": "${input:bridgeApiKey}",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "nvidia/meta/llama-3.1-8b-instruct",
        "name": "NVIDIA via Render Bridge",
        "url": "https://YOUR-SERVICE.onrender.com/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 32768,
        "maxOutputTokens": 8192
      }
    ]
  }
]
```

The `id` is sent as the request's `model` value. The bridge strips the `nvidia/` prefix before forwarding it to NVIDIA. For OpenRouter, add another model entry with an ID such as `openrouter/ACTUAL_MODEL_ID`; the bridge strips `openrouter/` and forwards the remainder.

For Agent Mode, enable the VS Code BYOK model-in-agent-host setting if your VS Code build requires it: `chat.agentHost.byokModels.enabled`. The model must support tool calling. Capability flags and context limits must reflect the actual upstream model, not assumptions.

## Local smoke test

Set `BRIDGE_API_KEY` and `NVIDIA_API_KEY`, then run:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Test (replace model with one enabled for your NVIDIA account):

```bash
curl -N https://YOUR-SERVICE.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_BRIDGE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"nvidia/meta/llama-3.1-8b-instruct",
    "stream":true,
    "messages":[{"role":"user","content":"Say hello briefly."}]
  }'
```

For OpenRouter, use `"model":"openrouter/OPENROUTER_MODEL_ID"`.

## Security

- Never commit API keys.
- Use HTTPS only.
- Use a long random bridge key.
- This is a personal proxy, not a public multi-user service.
- Consider adding rate limits and request-size limits before exposing it beyond your own devices.

## Latency testing

Compare direct provider vs bridge using the same model, prompt, region/time, and stream setting. Record:
- time to first byte/token
- total completion time
- tokens/second
- HTTP status and provider request ID

Run at least 10 trials and compare medians. A Render proxy can be slower than direct calls if its region is far from the provider.
