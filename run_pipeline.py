"""Command-line run of the whole pipeline (no web app needed).
Usage:  python run_pipeline.py [path/to/data.xlsx]"""
import sys
import pandas as pd
import data_loader, forecasting, risk_model

pd.set_option("display.width", 140)
path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/food_rescue_sample.xlsx"
df, rep = data_loader.load_dataset(path)
print("Cleaning report:", rep)
print(f"\n{len(df)} donations | {df.quantity_kg.sum():,.0f} kg | wastage rate {df.wasted.mean():.1%}")

s = forecasting.daily_series(df)
f = forecasting.train_forecaster(s)
print("\nForecast model comparison (time-based test set):")
print(f["results"].round(2).to_string(index=False))
print("Best ML model:", f["best_model_name"], "| beats baselines:", f["beats_baseline"])
print("\nNext 7 days (kg):")
print(forecasting.forecast_future(f["model"], s, 7).round(1).to_string())

r = risk_model.train_risk_model(df)
print("\nWastage-risk classifier:")
print({k: round(float(v), 3) for k, v in r["metrics"].items()})
print("Confusion:", {k: int(v) for k, v in r["confusion"].items()})
print("Feature importance:\n", r["importance"].round(3).to_string())
