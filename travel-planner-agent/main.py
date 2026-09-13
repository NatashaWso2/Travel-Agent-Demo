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
search_flights and search_hotels tools. Once you have picked one flight and one hotel,
reply with a final message that is ONLY a JSON object (no markdown, no extra text) shaped like:

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


def run_agent_loop(user_message: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

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

        return _parse_json_content(msg.content)

    raise RuntimeError("Planner did not converge on a recommendation")


def _parse_json_content(content: str | None) -> dict:
    if not content or not content.strip():
        raise RuntimeError("LLM returned empty content instead of the expected JSON")

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            text = rest if first_line.strip().lower() in ("", "json") else text

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse LLM response as JSON: {content!r}") from e


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    request_start = time.monotonic()
    plan = run_agent_loop(req.message)
    logger.info("run_agent_loop total: %.2fs", time.monotonic() - request_start)
    trip = plan["trip"]

    policy_start = time.monotonic()
    policy_result = requests.post(
        f"{TRAVEL_POLICY_URL}/check-policy",
        json={"trip": trip},
        headers={"x-api-key": TRAVEL_POLICY_API_KEY},
        timeout=15,
    ).json()
    logger.info("policy check took %.2fs", time.monotonic() - policy_start)

    lines = [plan["summary"], "", f"Policy decision: {policy_result['decision']}"]
    if policy_result["violations"]:
        lines.append("Violations:")
        lines += [f"- {v}" for v in policy_result["violations"]]

    return ChatResponse(response="\n".join(lines))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
