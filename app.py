"""Food Rescue Analytics: web dashboard.   Run:  streamlit run app.py"""
from pathlib import Path
import pandas as pd
import streamlit as st

import data_loader, forecasting, risk_model, config

st.set_page_config(page_title="Food Rescue Analytics", page_icon="🍲", layout="wide")
SAMPLE = Path(__file__).parent / "sample_data" / "food_rescue_sample.xlsx"


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_sample():
    return data_loader.load_dataset(str(SAMPLE))


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_uploaded(names, blobs):
    import io
    files = {n: io.BytesIO(b) for n, b in zip(names, blobs)}
    if len(files) == 1 and next(iter(files)).lower().endswith((".xlsx", ".xls")):
        return data_loader.load_dataset(next(iter(files.values())))
    return data_loader.load_dataset(csv_files=files)


@st.cache_resource(show_spinner="Training forecast models...")
def get_forecaster(df_key, category, _series):
    return forecasting.train_forecaster(_series)


@st.cache_resource(show_spinner="Training wastage-risk model...")
def get_risk_model(df_key, _df):
    return risk_model.train_risk_model(_df)


# ---------------- sidebar: data source
st.sidebar.title("🍲 Food Rescue Analytics")
source = st.sidebar.radio("Data source", ["Sample data (simulated)", "Upload my data"])
try:
    if source.startswith("Sample"):
        if not SAMPLE.exists():
            st.error("Run  python generate_sample_data.py  first to create the sample file.")
            st.stop()
        df, report = load_sample()
        st.sidebar.warning("Using SIMULATED sample data. Do not report these results as real findings.")
        key = "sample"
    else:
        ups = st.sidebar.file_uploader("Excel workbook (sheets: Donors, Donations, NGOs, Claims) or CSV files",
                                       type=["xlsx", "xls", "csv"], accept_multiple_files=True)
        if not ups:
            st.info("Upload your data in the sidebar to begin.")
            st.stop()
        df, report = load_uploaded(tuple(u.name for u in ups), tuple(u.getvalue() for u in ups))
        key = "|".join(u.name + str(u.size) for u in ups)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.title("Food Rescue Analytics: Surplus Food Prediction System")
tab_over, tab_fc, tab_risk, tab_data = st.tabs(["📊 Overview", "📈 Surplus forecast", "⚠️ Wastage risk", "🗂 Data"])

# ---------------- overview
with tab_over:
    c = st.columns(5)
    c[0].metric("Donations", f"{len(df):,}")
    c[1].metric("Total food (kg)", f"{df['quantity_kg'].sum():,.0f}")
    c[2].metric("Estimated meals", f"{df['estimated_meals'].sum():,.0f}")
    c[3].metric("Wastage rate", f"{df['wasted'].mean():.1%}")
    if "pickup_delay_min" in df.columns:
        c[4].metric("Avg pickup delay (min)", f"{df['pickup_delay_min'].mean():.0f}")
    st.subheader("Monthly donated quantity (kg)")
    monthly = df.set_index("donation_date")["quantity_kg"].resample("MS").sum()
    st.line_chart(monthly)
    a, b = st.columns(2)
    with a:
        st.subheader("Quantity by food category (kg)")
        st.bar_chart(df.groupby("food_category")["quantity_kg"].sum().sort_values(ascending=False))
    with b:
        st.subheader("Wastage rate by food category")
        st.bar_chart(df.groupby("food_category")["wasted"].mean().sort_values(ascending=False))
    if "donor_type" in df.columns:
        a, b = st.columns(2)
        with a:
            st.subheader("Quantity by donor type (kg)")
            st.bar_chart(df.groupby("donor_type")["quantity_kg"].sum().sort_values(ascending=False))
        with b:
            st.subheader("Wastage rate by donor type")
            st.bar_chart(df.groupby("donor_type")["wasted"].mean().sort_values(ascending=False))
    st.subheader("Average donation by weekday (kg per day)")
    daily = forecasting.daily_series(df)
    st.bar_chart(daily.groupby(daily.index.day_name()).mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]))

# ---------------- forecast
with tab_fc:
    st.write("Forecast how much surplus food (kg) will be donated each day. Models are trained on earlier "
             "data and tested on the most recent 20% of days, and compared with simple baselines.")
    cats = ["All"] + sorted(df["food_category"].dropna().unique().tolist())
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Food category", cats)
    days = c2.slider("Days to forecast", 7, 30, 14)
    try:
        series = forecasting.daily_series(df, cat)
        res = get_forecaster(key, cat, series)
        st.subheader("Model comparison on the test period")
        st.dataframe(res["results"].round(2), hide_index=True, use_container_width=True)
        if res["beats_baseline"]:
            st.success(f"Best model: {res['best_model_name']}. It beats both simple baselines on MAE.")
        else:
            st.warning(f"{res['best_model_name']} did not beat the simple baselines for this category. "
                       "Treat its forecast with caution (the data may be too small or irregular).")
        fut = forecasting.forecast_future(res["model"], series, days)
        view = pd.DataFrame({"Actual (last 60 days)": series.iloc[-60:]}).join(fut, how="outer")
        st.subheader(f"Forecast for the next {days} days")
        st.line_chart(view.rename(columns={"forecast_kg": "Forecast"}))
        st.dataframe(fut.round(1).rename_axis("date").reset_index(), hide_index=True)
        st.download_button("Download forecast (CSV)", fut.round(1).to_csv().encode(), "forecast.csv", "text/csv")
    except ValueError as e:
        st.error(str(e))

# ---------------- risk
with tab_risk:
    st.write("A classifier estimates the chance that a donation will be wasted, using only information known "
             "when it is donated (quantity, time to expiry, category, donor type, day and time).")
    try:
        rm = get_risk_model(key, df)
        m = rm["metrics"]
        c = st.columns(5)
        for col, name in zip(c, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
            col.metric(name, f"{m[name]:.2f}")
        st.caption(f"Accuracy of always predicting the majority class: "
                   f"{m['Baseline accuracy (always predict majority)']:.2f}. "
                   f"Wastage rate in the test period: {rm['test_wastage_rate']:.1%}.")
        a, b = st.columns(2)
        with a:
            st.subheader("What drives wastage risk")
            st.bar_chart(rm["importance"])
        with b:
            st.subheader("Confusion matrix (test period)")
            st.dataframe(pd.Series(rm["confusion"], name="count").to_frame(), use_container_width=True)
        st.subheader("Score recent donations")
        n = st.slider("Number of most recent donations to score", 20, 500, 100, step=20)
        recent = df.tail(n)
        scored = risk_model.score_donations(rm["model"], recent)
        level = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
        cols = [c for c in ["donation_id", "donation_date", "food_category", "quantity_kg", "hours_to_expiry",
                            "donor_type", "risk_probability", "risk_level"] if c in scored.columns]
        show = scored[scored["risk_level"].isin(level)][cols].sort_values("risk_probability", ascending=False)
        st.dataframe(show.round({"risk_probability": 2, "quantity_kg": 1, "hours_to_expiry": 1}),
                     hide_index=True, use_container_width=True)
        st.download_button("Download risk table (CSV)", show.to_csv(index=False).encode(), "risk_alerts.csv", "text/csv")
    except ValueError as e:
        st.error(str(e))

# ---------------- data
with tab_data:
    st.subheader("Cleaning report")
    st.json({k: int(v) for k, v in report.items()})
    st.caption("Wastage rule: a donation counts as wasted if its status says so, its delivery failed or expired, "
               "or it was never claimed. Change WASTE_WORDS in config.py to match your data.")
    st.subheader("Combined dataset (first 200 rows)")
    st.dataframe(df.head(200), use_container_width=True)
