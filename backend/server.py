from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os, json, logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

EMERGENT_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
OPENAI_BASE = "https://integrations.emergentagent.com/llm/openai/v1"

app = FastAPI(title="Dr. CV Voice Agent")
api = APIRouter(prefix="/api")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

log = logging.getLogger("drcv")


# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    system: str
    messages: List[ChatMessage]
    model: str = "gpt-4o"


class TTSRequest(BaseModel):
    text: str
    voice: str = "shimmer"
    speed: float = 1.05


class ExtractRequest(BaseModel):
    transcript: str
    existing: Dict[str, Any] = {}


# ─────────────────────────────────────────
# Health
# ─────────────────────────────────────────

@api.get("/")
async def root():
    return {
        "ok": True,
        "service": "drcv",
        "key_set": bool(EMERGENT_KEY)
    }


# ─────────────────────────────────────────
# STT
# ─────────────────────────────────────────

@api.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("en"),
    prompt: str = Form("")
):
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(400, "Empty audio")

    files = {
        "file": (
            file.filename or "audio.webm",
            audio_bytes,
            file.content_type or "audio/webm"
        )
    }

    data = {
        "model": "whisper-1",
        "language": language,
        "response_format": "json",
        "temperature": "0.0",
    }

    if prompt:
        data["prompt"] = prompt

    headers = {
        "Authorization": f"Bearer {EMERGENT_KEY}"
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(
                f"{OPENAI_BASE}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )

        if r.status_code != 200:
            log.error(f"Whisper error {r.status_code}: {r.text[:400]}")
            raise HTTPException(
                r.status_code,
                f"Transcribe failed: {r.text[:200]}"
            )

        result = r.json()

        return {
            "text": (result.get("text") or "").strip()
        }

    except httpx.HTTPError as e:
        log.exception("transcribe http err")
        raise HTTPException(502, f"Whisper unreachable: {e}")


# ─────────────────────────────────────────
# Chat Streaming
# ─────────────────────────────────────────

@api.post("/chat/stream")
async def chat_stream(req: ChatRequest):

    messages = [
        {"role": "system", "content": req.system}
    ] + [m.model_dump() for m in req.messages]

    body = {
        "model": req.model,
        "messages": messages,
        "stream": True,
        "temperature": 0.75,
        "max_tokens": 600,
    }

    headers = {
        "Authorization": f"Bearer {EMERGENT_KEY}",
        "Content-Type": "application/json",
    }

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OPENAI_BASE}/chat/completions",
                    headers=headers,
                    json=body,
                ) as r:

                    if r.status_code != 200:
                        err = await r.aread()

                        yield (
                            f"data: "
                            f"{json.dumps({'type':'error','message': err.decode('utf-8','ignore')[:200]})}\n\n"
                        )

                        return

                    async for line in r.aiter_lines():

                        if not line or not line.startswith("data:"):
                            continue

                        payload = line[5:].strip()

                        if payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return

                        try:
                            d = json.loads(payload)

                            delta = (
                                d.get("choices", [{}])[0]
                                .get("delta", {})
                            )

                            token = delta.get("content")

                            if token:
                                yield (
                                    f"data: "
                                    f"{json.dumps({'type':'token','text': token})}\n\n"
                                )

                        except Exception:
                            continue

        except Exception as e:
            log.exception("chat stream")

            yield (
                f"data: "
                f"{json.dumps({'type':'error','message': str(e)})}\n\n"
            )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────
# TTS
# ─────────────────────────────────────────

@api.post("/tts")
async def tts(req: TTSRequest):

    body = {
        "model": "tts-1",
        "input": req.text[:3500],
        "voice": req.voice,
        "speed": max(0.7, min(1.4, req.speed)),
        "response_format": "mp3",
    }

    headers = {
        "Authorization": f"Bearer {EMERGENT_KEY}",
        "Content-Type": "application/json",
    }

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:

                async with client.stream(
                    "POST",
                    f"{OPENAI_BASE}/audio/speech",
                    headers=headers,
                    json=body,
                ) as r:

                    if r.status_code != 200:
                        return

                    async for chunk in r.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk

        except Exception:
            log.exception("tts stream")

    return StreamingResponse(
        gen(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────
# Extract
# ─────────────────────────────────────────

@api.post("/extract")
async def extract(req: ExtractRequest):

    sys_p = """
Extract structured business info from the user's transcript.

Return STRICT JSON with these optional keys:
- goal
- timeline
- budget
- blocker
- decision_maker
- success
- industry

Return ONLY raw JSON.
"""

    user_p = f"Transcript: {req.transcript}"

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {EMERGENT_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:

            r = await client.post(
                f"{OPENAI_BASE}/chat/completions",
                headers=headers,
                json=body,
            )

        if r.status_code != 200:
            return {"extracted": {}}

        content = r.json()["choices"][0]["message"]["content"]

        try:
            data = json.loads(content)
        except Exception:
            data = {}

        return {"extracted": data}

    except Exception as e:
        log.warning(f"extract fail: {e}")
        return {"extracted": {}}


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
