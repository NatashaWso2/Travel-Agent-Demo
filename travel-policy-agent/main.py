from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Travel Policy Agent")

POLICY = {
    "max_budget_eur": 700,
    "allowed_cabin_classes": ["economy"],
    "latest_arrival_time": "10:00",
    "max_hotel_price_per_night_eur": 350,
}


class Trip(BaseModel):
    origin: str
    destination: str
    cabin_class: str
    arrival_time: str  # "HH:MM", 24h
    total_cost_eur: float
    hotel_price_per_night_eur: float


class PolicyRequest(BaseModel):
    trip: Trip


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-policy")
def check_policy(req: PolicyRequest):
    trip = req.trip
    violations = []

    if trip.total_cost_eur > POLICY["max_budget_eur"]:
        violations.append(
            f"Total cost EUR{trip.total_cost_eur} exceeds budget cap of "
            f"EUR{POLICY['max_budget_eur']}"
        )

    if trip.cabin_class.lower() not in POLICY["allowed_cabin_classes"]:
        violations.append(
            f"Cabin class '{trip.cabin_class}' is not allowed "
            f"(only {', '.join(POLICY['allowed_cabin_classes'])})"
        )

    if trip.arrival_time > POLICY["latest_arrival_time"]:
        violations.append(
            f"Arrival time {trip.arrival_time} is later than the "
            f"{POLICY['latest_arrival_time']} cutoff"
        )

    if trip.hotel_price_per_night_eur > POLICY["max_hotel_price_per_night_eur"]:
        violations.append(
            f"Hotel price EUR{trip.hotel_price_per_night_eur}/night exceeds cap of "
            f"EUR{POLICY['max_hotel_price_per_night_eur']}/night"
        )

    return {
        "decision": "APPROVED" if not violations else "REJECTED",
        "compliant": not violations,
        "violations": violations,
        "policy": POLICY,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
