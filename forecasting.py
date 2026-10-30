"""Module 3 and 4: feature engineering and surplus-quantity forecasting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

LAGS = (1, 2, 3, 7, 14)
ROLLS = (7, 14)


def daily_series(df, category=None):
    """Total surplus kg per day (days with no donation count as 0)."""
    d = df if category in (None, "All") else df[df["food_category"] == category]
    s = d.groupby(d["donation_date"].dt.normalize())["quantity_kg"].sum()
    idx = pd.date_range(df["donation_date"].min().normalize(), df["donation_date"].max().normalize())
    return s.reindex(idx, fill_value=0.0).rename("qty")


def make_features(s):
    X = pd.DataFrame(index=s.index)
    for lag in LAGS:
        X[f"lag_{lag}"] = s.shift(lag)
    for r in ROLLS:
        X[f"roll_{r}"] = s.shift(1).rolling(r).mean()
    X["dow"] = s.index.dayofweek
    X["is_weekend"] = (s.index.dayofweek >= 5).astype(int)
    X["month"] = s.index.month
    X["day_of_month"] = s.index.day
    return X


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    nz = y > 0
    mape = np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100 if nz.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE %": mape}


def train_forecaster(series, test_frac=0.2):
    """Train on the earlier part, test on the latest part (time-based split).
    Compares two simple baselines with two ML models and returns the results."""
    if len(series) < 60:
        raise ValueError("Need at least 60 days of data to train a forecast model.")
    X = make_features(series)
    data = X.assign(y=series).dropna()
    cut = int(len(data) * (1 - test_frac))
    tr, te = data.iloc[:cut], data.iloc[cut:]
    Xtr, ytr, Xte, yte = tr.drop(columns="y"), tr["y"], te.drop(columns="y"), te["y"]
    rows, preds = [], {}
    preds["Baseline: same day last week"] = Xte["lag_7"]
    preds["Baseline: 7-day moving average"] = Xte["roll_7"]
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    fitted = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds[name] = m.predict(Xte)
        fitted[name] = m
    for name, p in preds.items():
        rows.append({"Model": name, **_metrics(yte, p)})
    res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_ml = min(models, key=lambda n: res.loc[res.Model == n, "MAE"].iloc[0])
    final = models[best_ml].__class__(**models[best_ml].get_params()).fit(data.drop(columns="y"), data["y"])
    best_base = res[res.Model.str.startswith("Baseline")]["MAE"].min()
    ml_mae = res.loc[res.Model == best_ml, "MAE"].iloc[0]
    return {"results": res, "best_model_name": best_ml, "model": final,
            "beats_baseline": bool(ml_mae < best_base), "series": series,
            "test_actual": yte, "test_pred": pd.Series(preds[best_ml], index=yte.index)}


def forecast_future(model, series, days=14):
    """Recursive forecast: each predicted day is used to build the next day's lags."""
    s = series.copy()
    out = {}
    for _ in range(days):
        nxt = s.index[-1] + pd.Timedelta(days=1)
        s.loc[nxt] = np.nan
        X = make_features(s).iloc[[-1]]
        val = float(max(model.predict(X)[0], 0))
        s.loc[nxt] = val
        out[nxt] = val
    return pd.Series(out, name="forecast_kg")
