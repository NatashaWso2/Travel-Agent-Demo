import json
import logging
import os
import time

import requests
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel-planner")

app = FastAPI(title="Travel Planner Agent")

# LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
GW_OPENAI_URL = os.environ.get("GW_OPENAI_URL")
# LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TRAVEL_POLICY_URL = os.environ.get("TRAVEL_POLICY_URL")
TRAVEL_POLICY_API_KEY = os.environ.get("TRAVEL_POLICY_API_KEY")

client = OpenAI(
    base_url=GW_OPENAI_URL,
    api_key="unused",
    timeout=20.0,
    max_retries=0,
)


# --- Mock flight/hotel tools -------------------------------------------------

def search_flights(origin: str, destination: str, date: str):
    return [
        {
            "flight_no": "AB123",
            "origin": origin,
            "destination": destination,
            "date": date,
            "cabin_class": "economy",
            "arrival_time": "09:15",
            "price_eur": 280,
        },
        {
            "flight_no": "CD456",
            "origin": origin,
            "destination": destination,
            "date": date,
            "cabin_class": "business",
            "arrival_time": "08:40",
            "price_eur": 950,
        },
        {
            "flight_no": "EF789",
            "origin": origin,
            "destination": destination,
            "date": date,
            "cabin_class": "economy",
            "arrival_time": "11:30",
            "price_eur": 240,
        },
    ]


def search_hotels(city: str, checkin: str, checkout: str):
    return [
        {"name": "City Center Inn", "city": city, "price_per_night_eur": 190},
        {"name": "Grand Plaza", "city": city, "price_per_night_eur": 420},
        {"name": "Budget Stay", "city": city, "price_per_night_eur": 110},
    ]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two cities on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search available hotels in a city for a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "checkin": {"type": "string", "description": "YYYY-MM-DD"},
                    "checkout": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["city", "checkin", "checkout"],
            },
        },
    },
]

TOOL_IMPL = {"search_flights": search_flights, "search_hotels": search_hotels}

SYSTEM_PROMPT = """You are the Travel Planner Agent. Help the user plan a trip using the
search_flights and search_hotels tools.

If you don't yet have enough information to search (e.g. origin, destination, or travel
dates are missing), or the user is just greeting you / asking a general question, reply
normally in plain text and ask for whatever is missing. Do not use the JSON format below
until you are actually proposing a concrete trip.

Once you have picked one flight and one hotel, reply with a final message that is ONLY a
JSON object (no markdown, no extra text) shaped like:

{
  "summary": "<one paragraph human-readable recommendation>",
  "trip": {
    "origin": "...",
    "destination": "...",
    "cabin_class": "economy|business|...",
    "arrival_time": "HH:MM",
    "total_cost_eur": <flight price + one night hotel price>,
    "hotel_price_per_night_eur": <number>
  }
}

Do not include any other commentary in that final message.
"""


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    context: dict | None = None


class ChatResponse(BaseModel):
    response: str


def run_agent_loop(messages: list) -> dict | str:
    """Drives the tool-calling loop on the given (mutated in place) message history.
    Returns a {summary, trip} dict once the model proposes a concrete trip, or a plain
    string if the model is instead asking a clarifying question / just chatting."""
    for step in range(3):
        step_start = time.monotonic()
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        logger.info("LLM call (step %d) took %.2fs", step, time.monotonic() - step_start)
        msg = completion.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                fn = TOOL_IMPL[call.function.name]
                args_str = (call.function.arguments or "").strip()
                args = json.loads(args_str) if args_str else {}
                result = fn(**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            continue

        messages.append({"role": "assistant", "content": msg.content})
        plan = _try_parse_trip_json(msg.content)
        return plan if plan is not None else msg.content

    raise RuntimeError("Planner did not converge on a recommendation")


def check_policy(trip: dict) -> dict:
    return requests.post(
        f"{TRAVEL_POLICY_URL}/check-policy",
        json={"trip": trip},
        headers={"x-api-key": TRAVEL_POLICY_API_KEY},
        timeout=15,
    ).json()


# In-memory conversation history per session_id. Fine for a demo with a single replica;
# a real deployment would back this with a shared store (redis, db, etc).
SESSIONS: dict[str, list] = {}


def get_session_messages(session_id: str | None) -> list:
    key = session_id or "default"
    if key not in SESSIONS:
        SESSIONS[key] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return SESSIONS[key]


def plan_trip(messages: list, user_message: str):
    """Returns (plan, policy_result).
    - If the model replied conversationally (no concrete trip yet), plan is that plain
      string and policy_result is None -- caller should return it as-is.
    - Otherwise plan is a {summary, trip} dict and policy_result is the compliance check."""
    messages.append({"role": "user", "content": user_message})

    plan = run_agent_loop(messages)
    if not isinstance(plan, dict):
        return plan, None

    return plan, check_policy(plan["trip"])


def _try_parse_trip_json(content: str | None) -> dict | None:
    """Best-effort parse of a {summary, trip} plan. Returns None (not an error) if the
    content isn't that shape -- e.g. the model is just chatting or asking a clarifying
    question, which is expected for messages like "hi" or "help me plan a trip"."""
    if not content or not content.strip():
        return None

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            text = rest if first_line.strip().lower() in ("", "json") else text

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or "trip" not in parsed or "summary" not in parsed:
        return None

    return parsed


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    request_start = time.monotonic()
    messages = get_session_messages(req.session_id)
    plan, policy_result = plan_trip(messages, req.message)
    logger.info("full request handling took %.2fs", time.monotonic() - request_start)

    if policy_result is None:
        # The model just replied conversationally (e.g. a greeting, or it needs more
        # details before it can propose a trip) -- pass that straight through.
        return ChatResponse(response=plan)

    if policy_result["compliant"]:
        return ChatResponse(
            response=f"{plan['summary']}\n\nThis trip is compliant with company travel policy."
        )

    violation_lines = [f"- {v}" for v in policy_result["violations"]]
    lines = [
        "What you asked for isn't allowed under company travel policy:",
        *violation_lines,
    ]
    return ChatResponse(response="\n".join(lines))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
