import asyncio
import json
import logging
import os
import time

import requests
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
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

# Injected by the "Configure MCP Tools" binding on this agent in the console.
FLIGHT_HOTEL_TOOLS_URL = os.environ.get("FLIGHT_HOTEL_TOOLS_URL")

client = OpenAI(
    base_url=GW_OPENAI_URL,
    api_key="unused",
    timeout=20.0,
    max_retries=0,
)


# --- Flight/hotel tools, called through the MCP proxy ------------------------

def call_mcp_tool(name: str, arguments: dict) -> list:
    async def _call():
        async with streamablehttp_client(FLIGHT_HOTEL_TOOLS_URL) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                start = time.monotonic()
                result = await session.call_tool(name, arguments)
                logger.info("MCP tool %s took %.2fs", name, time.monotonic() - start)
                return [
                    json.loads(block.text)
                    for block in result.content
                    if block.type == "text"
                ]

    return asyncio.run(_call())


def search_flights(origin: str, destination: str, date: str):
    return call_mcp_tool(
        "search_flights", {"origin": origin, "destination": destination, "date": date}
    )


def search_hotels(city: str, checkin: str, checkout: str):
    return call_mcp_tool(
        "search_hotels", {"city": city, "checkin": checkin, "checkout": checkout}
    )


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
    "flight_no": "...",
    "origin": "...",
    "destination": "...",
    "cabin_class": "economy|business|...",
    "arrival_time": "HH:MM",
    "flight_price_eur": <number>,
    "hotel_name": "...",
    "hotel_price_per_night_eur": <number>,
    "total_cost_eur": <flight price + one night hotel price>
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
    """Best-effort extraction of a {summary, trip} plan from the model's reply. Returns
    None (not an error) if no such object is found -- e.g. the model is just chatting or
    asking a clarifying question, which is expected for messages like "hi". Tolerates the
    model wrapping the JSON in markdown, or adding commentary before/after it -- despite
    the system prompt asking for JSON only, models don't always follow that strictly."""
    if not content or not content.strip():
        return None

    decoder = json.JSONDecoder()
    search_start = 0
    while True:
        brace_pos = content.find("{", search_start)
        if brace_pos == -1:
            return None
        try:
            parsed, end_pos = decoder.raw_decode(content, brace_pos)
        except json.JSONDecodeError:
            search_start = brace_pos + 1
            continue

        if isinstance(parsed, dict) and "trip" in parsed and "summary" in parsed:
            return parsed
        search_start = end_pos


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
        body = ["Within company travel policy. Ticket booked -- have a good trip."]
        return ChatResponse(response=_render_card("APPROVED", plan["trip"], body))

    violations = policy_result["violations"]
    suggestions = _suggestions_for(violations, plan["trip"])
    body = ["Doesn't clear company travel policy:"]
    body += [f"  x {v['message']}" for v in violations]
    body.append("")
    body.append("To get this approved:")
    body += [f"  > {s}" for s in suggestions]
    return ChatResponse(response=_render_card("REJECTED", plan["trip"], body))


def _render_card(status: str, trip: dict, body: list[str]) -> str:
    """One consistent 'ticket stub' shape for every trip outcome -- same header, same
    field layout, same rule -- so the reply reads as one system rather than a different
    format every time."""
    rule = "-" * 46
    flight_no = trip.get("flight_no") or "-"
    cabin = str(trip.get("cabin_class", "")).title()
    hotel_name = trip.get("hotel_name")
    hotel_price = _fmt_eur(trip.get("hotel_price_per_night_eur"))
    hotel_line = f"{hotel_name} ({hotel_price}/night)" if hotel_name else f"{hotel_price}/night"

    lines = [
        rule,
        f" TRAVEL PLANNER -- BOOKING {status}",
        rule,
        f" Flight   {flight_no}  ({trip.get('origin')} -> {trip.get('destination')})",
        f" Class    {cabin}",
        f" Arrives  {trip.get('arrival_time')}",
    ]
    if trip.get("flight_price_eur") is not None:
        lines.append(f" Fare     {_fmt_eur(trip['flight_price_eur'])}")
    lines.append(f" Hotel    {hotel_line}")
    lines.append(f" Total    {_fmt_eur(trip.get('total_cost_eur'))}")
    lines.append(rule)
    lines += [f" {b}" if b else "" for b in body]
    lines.append(rule)
    return "\n".join(lines)


RULE_SUGGESTIONS = {
    "budget": "Bring the total cost under {policy[max_budget_eur]} -- try a cheaper flight or hotel.",
    "cabin_class": "Book an economy fare instead of {trip[cabin_class]}.",
    "arrival_time": "Look for a flight that arrives before {policy[latest_arrival_time]}.",
    "hotel_price": "Choose a hotel at or under {policy[max_hotel_price_per_night_eur]} per night.",
}


def _suggestions_for(violations: list[dict], trip: dict) -> list[str]:
    policy = {
        "max_budget_eur": _fmt_eur(700),
        "latest_arrival_time": "10:00",
        "max_hotel_price_per_night_eur": _fmt_eur(350),
    }
    lines = []
    for v in violations:
        template = RULE_SUGGESTIONS.get(v["rule"])
        if template:
            lines.append(f"- {template.format(policy=policy, trip=trip)}")
    return lines or ["- Adjust the trip to fit company travel policy and ask again."]


def _fmt_eur(value) -> str:
    try:
        return f"€{float(value):,.0f}"
    except (TypeError, ValueError):
        return f"€{value}"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
