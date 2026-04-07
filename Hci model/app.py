import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# =========================
# Load environment
# =========================
load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_MODEL = os.getenv("ARK_MODEL", "deepseek-v3-2-251201")

if not ARK_API_KEY:
    raise RuntimeError("ARK_API_KEY is not set in .env")

client = OpenAI(
    api_key=ARK_API_KEY,
    base_url=ARK_BASE_URL,
)

# =========================
# App setup
# =========================
app = FastAPI(title="Adaptive Tutor Backend")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# =========================
# Data models
# =========================
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_input: str = Field(..., min_length=1)
    history: List[ChatMessage] = []
    response_time_seconds: Optional[float] = None
    fixed_mode: Optional[Literal["teacher", "friend", "adaptive"]] = "adaptive"
    use_web_search: Optional[bool] = False


class ChatResponse(BaseModel):
    session_id: str
    selected_role: Literal["teacher", "friend"]
    detected_state: str
    assistant_reply: str
    switched: bool


# =========================
# Prompt templates
# =========================
TEACHER_PROMPT = """
You are a supportive teacher for children aged 7-12.

Your job:
- guide the child with short, clear, structured questions
- do not directly give the final answer unless absolutely necessary
- help the child think step by step
- gently bring the child back if they are off-topic
- encourage reasoning, comparison, and reflection
- sound warm, but more structured than a friend

Style rules:
- use simple language
- keep replies concise
- ask at most one or two focused questions
- avoid long lectures
- do not overwhelm the child
"""

FRIEND_PROMPT = """
You are a friendly peer-like AI companion for children aged 7-12.

Your job:
- encourage the child
- make them feel safe and confident
- support idea exploration and creative thinking
- keep the conversation warm, playful, and low-pressure
- do not sound judgmental or evaluative

Style rules:
- use simple language
- keep replies concise
- praise effort
- invite the child to keep exploring
- do not over-correct too quickly
"""

# =========================
# Helper functions
# =========================
def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def is_stuck_phrase(text: str) -> bool:
    t = normalize_text(text)
    stuck_markers = [
        "i don't know",
        "i dont know",
        "i am stuck",
        "i'm stuck",
        "help me",
        "what should i do",
        "i can't do this",
        "cant do this",
        "no idea",
        "i am confused",
        "i'm confused",
    ]
    return any(marker in t for marker in stuck_markers)


def is_too_short(text: str) -> bool:
    return len(text.strip().split()) <= 2


def is_repeated_idea(user_text: str, history: List[ChatMessage]) -> bool:
    normalized = normalize_text(user_text)
    previous_user_msgs = [m.content for m in history if m.role == "user"]

    if not previous_user_msgs:
        return False

    last_user = normalize_text(previous_user_msgs[-1])
    return normalized == last_user and len(normalized) > 0


def is_off_topic(text: str) -> bool:
    t = normalize_text(text)
    obvious_offtopic_markers = [
        "minecraft",
        "roblox",
        "tiktok",
        "youtube",
        "my dog",
        "random",
    ]
    return any(marker in t for marker in obvious_offtopic_markers)


def is_productive_exploration(text: str) -> bool:
    t = normalize_text(text)
    exploration_markers = [
        "maybe",
        "what if",
        "i think",
        "another idea",
        "for example",
        "because",
        "we could",
        "perhaps",
    ]
    return any(marker in t for marker in exploration_markers) and len(t.split()) >= 5


def detect_state(
    user_text: str,
    history: List[ChatMessage],
    response_time_seconds: Optional[float],
) -> str:
    """
    Returns one of:
    - stuck
    - off_topic
    - repetition
    - productive_exploration
    - neutral
    """
    if is_stuck_phrase(user_text):
        return "stuck"

    if is_off_topic(user_text):
        return "off_topic"

    if is_repeated_idea(user_text, history):
        return "repetition"

    if response_time_seconds is not None and response_time_seconds > 45:
        return "stuck"

    if is_too_short(user_text):
        return "stuck"

    if is_productive_exploration(user_text):
        return "productive_exploration"

    return "neutral"


def choose_role(
    detected_state: str,
    fixed_mode: Literal["teacher", "friend", "adaptive"],
) -> Literal["teacher", "friend"]:
    if fixed_mode == "teacher":
        return "teacher"

    if fixed_mode == "friend":
        return "friend"

    # adaptive mode
    if detected_state in {"stuck", "off_topic", "repetition"}:
        return "teacher"

    return "friend"


def build_instructions(selected_role: Literal["teacher", "friend"]) -> str:
    return TEACHER_PROMPT if selected_role == "teacher" else FRIEND_PROMPT


def build_conversation_text(history: List[ChatMessage], user_input: str) -> str:
    """
    Convert multi-turn history into one plain-text conversation block.
    This is the safest cross-provider format for OpenAI-compatible APIs.
    """
    parts = []
    for msg in history:
        parts.append(f"{msg.role.upper()}: {msg.content}")
    parts.append(f"USER: {user_input}")
    return "\n".join(parts)


def extract_response_text(response) -> str:
    """
    Try to extract generated text from different compatible response shapes.
    """
    # Preferred
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    # Fallback: parse output content manually
    if hasattr(response, "output") and response.output:
        collected = []
        for item in response.output:
            content_list = getattr(item, "content", None)
            if not content_list:
                continue
            for c in content_list:
                c_type = getattr(c, "type", None)
                if c_type in ("output_text", "text"):
                    text_value = getattr(c, "text", None)
                    if text_value:
                        collected.append(text_value)
        if collected:
            return "\n".join(collected).strip()

    # Final fallback
    return "Sorry, I could not generate a response."


def call_llm(
    selected_role: Literal["teacher", "friend"],
    history: List[ChatMessage],
    user_input: str,
    use_web_search: bool = False,
) -> str:
    instructions = build_instructions(selected_role)
    full_text = build_conversation_text(history, user_input)

    request_kwargs = {
        "model": ARK_MODEL,
        "stream": False,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": full_text
                    }
                ]
            }
        ],
    }

    if use_web_search:
        request_kwargs["tools"] = [
            {
                "type": "web_search",
                "max_keyword": 3
            }
        ]

    try:
        response = client.responses.create(**request_kwargs)
        return extract_response_text(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")


def save_log(data: dict):
    session_id = data["session_id"]
    log_file = LOG_DIR / f"{session_id}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {
        "message": "Adaptive Tutor Backend is running.",
        "model": ARK_MODEL,
        "base_url": ARK_BASE_URL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    detected_state = detect_state(
        user_text=req.user_input,
        history=req.history,
        response_time_seconds=req.response_time_seconds,
    )

    selected_role = choose_role(
        detected_state=detected_state,
        fixed_mode=req.fixed_mode,
    )

    assistant_reply = call_llm(
        selected_role=selected_role,
        history=req.history,
        user_input=req.user_input,
        use_web_search=req.use_web_search,
    )

    switched = (
        req.fixed_mode == "adaptive"
        and detected_state in {"stuck", "off_topic", "repetition"}
    )

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_input": req.user_input,
        "history_length": len(req.history),
        "response_time_seconds": req.response_time_seconds,
        "fixed_mode": req.fixed_mode,
        "use_web_search": req.use_web_search,
        "detected_state": detected_state,
        "selected_role": selected_role,
        "assistant_reply": assistant_reply,
        "switched": switched,
        "model": ARK_MODEL,
        "base_url": ARK_BASE_URL,
    }
    save_log(log_record)

    return ChatResponse(
        session_id=session_id,
        selected_role=selected_role,
        detected_state=detected_state,
        assistant_reply=assistant_reply,
        switched=switched,
    )