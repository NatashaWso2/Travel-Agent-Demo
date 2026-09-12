import asyncio
import json
import os

import requests
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Travel Planner Agent")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TRAVEL_POLICY_URL = os.environ.get("TRAVEL_POLICY_URL")

# Injected by the "MCP" binding on this agent in the console (you choose these
# env var names when you click + Add under Configure MCP Tools).
MCP_TOOLS_URL = os.environ.get("MCP_TOOLS_URL")
MCP_TOOLS_API_KEY = os.environ.get("MCP_TOOLS_API_KEY")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# --- Real flight/hotel tools, called through the MCP proxy -------------------

def call_mcp_tool(name: str, arguments: dict) -> list:
    async def _call():
        headers = {"Authorization": f"Bearer {MCP_TOOLS_API_KEY}"}
        async with streamablehttp_client(MCP_TOOLS_URL, headers=headers) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
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

    for _ in range(5):
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = completion.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                fn = TOOL_IMPL[call.function.name]
                args = json.loads(call.function.arguments or "{}")
                result = fn(**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            continue

        return json.loads(msg.content)

    raise RuntimeError("Planner did not converge on a recommendation")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    plan = run_agent_loop(req.message)
    trip = plan["trip"]

    policy_result = requests.post(
        f"{TRAVEL_POLICY_URL}/check-policy", json={"trip": trip}, timeout=15
    ).json()

    lines = [plan["summary"], "", f"Policy decision: {policy_result['decision']}"]
    if policy_result["violations"]:
        lines.append("Violations:")
        lines += [f"- {v}" for v in policy_result["violations"]]

    return ChatResponse(response="\n".join(lines))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
