from mcp.server.fastmcp import FastMCP

mcp = FastMCP("flight-hotel-tools", host="0.0.0.0", port=8000)


@mcp.tool()
def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Search available flights between two cities on a date (YYYY-MM-DD)."""
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


@mcp.tool()
def search_hotels(city: str, checkin: str, checkout: str) -> list[dict]:
    """Search available hotels in a city for a check-in/check-out date range (YYYY-MM-DD)."""
    return [
        {"name": "City Center Inn", "city": city, "price_per_night_eur": 190},
        {"name": "Grand Plaza", "city": city, "price_per_night_eur": 420},
        {"name": "Budget Stay", "city": city, "price_per_night_eur": 110},
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
