"""Creates a SIMULATED dataset (donors, donations, NGOs, claims) so the whole
pipeline can be run and tested before the real data is plugged in.

The numbers are random but follow realistic rules (weekend peaks, festival
season, short shelf life of cooked food ...). Do NOT report results from this
data as real findings.   Run:  python generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "sample_data" / "food_rescue_sample.xlsx"

DONOR_TYPES = {  # type: (typical kg, share of donors)
    "Restaurant": (14, 0.28), "Caterer": (45, 0.16), "Hotel": (32, 0.14),
    "Event Organiser": (60, 0.12), "Grocery Store": (22, 0.18), "Household": (4, 0.12),
}
CATEGORIES = {  # category: (min shelf-life h, max shelf-life h)
    "Cooked Meals": (5, 10), "Bakery": (24, 48), "Fruits & Vegetables": (48, 96),
    "Packaged Food": (240, 720), "Dairy": (36, 72),
}
CAT_MIX = {  # category probabilities per donor type
    "Restaurant": [0.7, 0.1, 0.05, 0.05, 0.1], "Caterer": [0.85, 0.05, 0.03, 0.02, 0.05],
    "Hotel": [0.6, 0.15, 0.1, 0.05, 0.1], "Event Organiser": [0.8, 0.08, 0.04, 0.03, 0.05],
    "Grocery Store": [0.05, 0.15, 0.4, 0.25, 0.15], "Household": [0.6, 0.15, 0.1, 0.05, 0.1],
}
CITIES = ["Delhi", "Noida", "Greater Noida", "Ghaziabad", "Gurugram"]

# ---- donors and NGOs
n_donors, n_ngos = 40, 12
types = rng.choice(list(DONOR_TYPES), n_donors, p=[v[1] for v in DONOR_TYPES.values()])
donors = pd.DataFrame({"donor_id": [f"D{i:03d}" for i in range(1, n_donors + 1)],
                       "name": [f"Donor {i}" for i in range(1, n_donors + 1)],
                       "donor_type": types, "city": rng.choice(CITIES, n_donors)})
ngos = pd.DataFrame({"ngo_id": [f"N{i:02d}" for i in range(1, n_ngos + 1)],
                     "name": [f"NGO {i}" for i in range(1, n_ngos + 1)],
                     "city": rng.choice(CITIES, n_ngos),
                     "daily_demand_kg": rng.integers(40, 160, n_ngos)})

# ---- donations
rows = []
days = pd.date_range("2025-01-01", "2026-10-15")
did = 1
for day in days:
    dow = day.dayofweek
    base = 7 + (4 if dow >= 5 else 0) + (2 if dow == 4 else 0)
    if day.month in (10, 11): base *= 1.35      # festival season
    if day.month == 12 or (day.month == 2): base *= 1.2   # wedding season
    for _ in range(rng.poisson(base)):
        d = donors.iloc[rng.integers(n_donors)]
        cat = rng.choice(list(CATEGORIES), p=CAT_MIX[d.donor_type])
        kg = float(np.round(rng.lognormal(np.log(DONOR_TYPES[d.donor_type][0]), 0.5), 1))
        prep = day + pd.Timedelta(hours=int(rng.integers(8, 22)), minutes=int(rng.integers(0, 60)))
        lo, hi = CATEGORIES[cat]
        expiry = prep + pd.Timedelta(hours=float(rng.uniform(lo, hi)))
        rows.append((f"F{did:06d}", d.donor_id, day, cat, kg, prep, expiry))
        did += 1
don = pd.DataFrame(rows, columns=["donation_id", "donor_id", "donation_date", "food_category",
                                  "quantity_kg", "prepared_time", "expiry_time"])
don = don.merge(donors[["donor_id", "donor_type"]], on="donor_id")
don["hours_left"] = (don.expiry_time - don.prepared_time).dt.total_seconds() / 3600

# ---- claims: harder to claim when time is short, quantity is large or on weekends
p_claim = (0.93 - 0.35 * (don.hours_left < 8) - 0.0035 * don.quantity_kg.clip(upper=100)
           - 0.06 * (don.donation_date.dt.dayofweek >= 5) - 0.10 * (don.donor_type == "Household"))
p_claim = p_claim.clip(0.25, 0.97)
claimed = rng.random(len(don)) < p_claim
cl = don[claimed].copy()
cl["claim_id"] = [f"C{i:06d}" for i in range(1, len(cl) + 1)]
cl["ngo_id"] = rng.choice(ngos.ngo_id, len(cl))
cl["claim_time"] = cl.prepared_time + pd.to_timedelta(rng.uniform(0.3, 4, len(cl)), unit="h")
cl["pickup_distance_km"] = np.round(rng.gamma(2.5, 3.0, len(cl)), 1)
cl["pickup_delay_min"] = np.round(rng.gamma(2.0, 12, len(cl)) + 2 * cl.pickup_distance_km).astype(int)
pickup = cl.claim_time + pd.to_timedelta(cl.pickup_delay_min, unit="m")
late = pickup > cl.expiry_time - pd.Timedelta(hours=0.5)
cl["delivery_status"] = np.where(late, "Expired", "Delivered")
claims = cl[["claim_id", "donation_id", "ngo_id", "claim_time", "pickup_distance_km",
             "pickup_delay_min", "delivery_status"]]

don["status"] = "Wasted"
ok = claims[claims.delivery_status == "Delivered"].donation_id
don.loc[don.donation_id.isin(ok), "status"] = "Delivered"
don["estimated_meals"] = (don.quantity_kg * 2.5).round().astype(int)
donations = don[["donation_id", "donor_id", "donation_date", "food_category", "quantity_kg",
                 "estimated_meals", "prepared_time", "expiry_time", "status"]]

OUT.parent.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT) as xw:
    donors.to_excel(xw, sheet_name="Donors", index=False); donations.to_excel(xw, sheet_name="Donations", index=False)
    ngos.to_excel(xw, sheet_name="NGOs", index=False); claims.to_excel(xw, sheet_name="Claims", index=False)
print(f"Saved {OUT}: {len(donors)} donors, {len(donations)} donations, {len(ngos)} NGOs, {len(claims)} claims")
