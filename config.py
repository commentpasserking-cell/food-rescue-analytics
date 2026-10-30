"""Central settings for the Food Rescue Analytics project.

If your real dataset uses different column names or different words for
'wasted', change ALIASES / WASTE_WORDS here: no other file needs editing.
"""

# Assumption used only when the data has no 'estimated_meals' column.
MEALS_PER_KG = 2.5

# Values (lower-case) in a status column that mean the food was NOT used.
WASTE_WORDS = {
    "wasted", "expired", "spoiled", "discarded", "rejected", "failed",
    "cancelled", "canceled", "not delivered", "undelivered",
}

# Probability limits for Low / Medium / High wastage-risk labels.
RISK_BINS = (0.33, 0.66)

# Sheet / file name keywords -> table. First match wins, so order matters.
TABLE_KEYWORDS = [
    ("claims", ["claim"]),
    ("ngos", ["ngo", "receiver", "recipient", "charity"]),
    ("donors", ["donor", "provider"]),
    ("donations", ["donation", "food_listing", "listing", "surplus"]),
]

# Column-name aliases (all compared in lower_snake_case) -> canonical name.
ALIASES = {
    "donation_id": ["donation_id", "donationid", "food_id", "listing_id", "food_donation_id"],
    "donor_id": ["donor_id", "donorid", "provider_id"],
    "ngo_id": ["ngo_id", "ngoid", "receiver_id", "recipient_id"],
    "claim_id": ["claim_id", "claimid"],
    "donation_date": ["donation_date", "date", "date_donated", "donated_on", "listing_date", "created_at"],
    "food_category": ["food_category", "category", "food_type", "type_of_food"],
    "quantity_kg": ["quantity_kg", "quantity", "qty", "food_quantity", "quantity_(kg)", "weight_kg"],
    "prepared_time": ["prepared_time", "prepared_at", "preparation_time", "cooked_time"],
    "expiry_time": ["expiry_time", "expiry_date", "expiry", "expires_at", "best_before", "expiration_date"],
    "status": ["status", "donation_status"],
    "donor_type": ["donor_type", "provider_type"],
    "city": ["city", "location", "area"],
    "name": ["name", "donor_name", "ngo_name", "provider_name"],
    "claim_time": ["claim_time", "claimed_at", "claim_date", "date_claimed"],
    "pickup_distance_km": ["pickup_distance_km", "distance_km", "distance", "pickup_distance"],
    "pickup_delay_min": ["pickup_delay_min", "delay_min", "pickup_delay", "delay_minutes"],
    "delivery_status": ["delivery_status", "claim_status"],
    "daily_demand_kg": ["daily_demand_kg", "demand_kg", "capacity_kg"],
    "estimated_meals": ["estimated_meals", "meals", "meals_served"],
}

# Table-specific overrides (applied first).
TABLE_ALIASES = {
    "claims": {"status": "delivery_status", "date": "claim_time"},
}

REQUIRED_DONATION_COLUMNS = ["donation_date", "quantity_kg"]
