"""Module 5: classify each donation as low / medium / high risk of being wasted."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import config

# Only information known when the donation is made (no claim details: that would be leakage).


def _prepare(df):
    X = pd.DataFrame(index=df.index)
    X["quantity_kg"] = df["quantity_kg"]
    if "hours_to_expiry" in df.columns:
        X["hours_to_expiry"] = df["hours_to_expiry"]
    t = df["prepared_time"] if "prepared_time" in df.columns else df["donation_date"]
    X["prep_hour"] = t.dt.hour
    X["dow"] = df["donation_date"].dt.dayofweek
    X["month"] = df["donation_date"].dt.month
    X["food_category"] = df["food_category"].astype("string").fillna("Unknown").astype(str)
    if "donor_type" in df.columns:
        X["donor_type"] = df["donor_type"].astype("string").fillna("Unknown").astype(str)
    if "donor_city" in df.columns and df["donor_city"].nunique() <= 30:
        X["donor_city"] = df["donor_city"].astype("string").fillna("Unknown").astype(str)
    return X


def train_risk_model(df, test_frac=0.2):
    if df["wasted"].nunique() < 2:
        raise ValueError("The data has only one outcome (all wasted or none wasted): cannot train a classifier.")
    df = df.sort_values("donation_date")
    X, y = _prepare(df), df["wasted"]
    cut = int(len(df) * (1 - test_frac))
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    cat = [c for c in ('food_category', 'donor_type', 'donor_city') if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    clf = Pipeline([("pre", pre), ("rf", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    major = int(ytr.mode().iloc[0])
    metrics = {
        "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred, zero_division=0), "F1-score": f1_score(yte, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(yte, proba) if yte.nunique() == 2 else np.nan,
        "Baseline accuracy (always predict majority)": accuracy_score(yte, np.full(len(yte), major)),
    }
    names = clf.named_steps["pre"].get_feature_names_out()
    imp = clf.named_steps["rf"].feature_importances_
    grouped = {}
    for n, v in zip(names, imp):
        base = n.split("__", 1)[1]
        for c in cat:
            if base.startswith(c + "_"):
                base = c
        grouped[base] = grouped.get(base, 0) + v
    importance = pd.Series(grouped).sort_values(ascending=False)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    return {"model": clf, "metrics": metrics, "importance": importance,
            "confusion": {"True negative": tn, "False positive": fp, "False negative": fn, "True positive": tp},
            "test_wastage_rate": float(yte.mean())}


def score_donations(model, df):
    out = df.copy()
    out["risk_probability"] = model.predict_proba(_prepare(df))[:, 1]
    lo, hi = config.RISK_BINS
    out["risk_level"] = pd.cut(out["risk_probability"], [-0.01, lo, hi, 1.01], labels=["Low", "Medium", "High"])
    return out
