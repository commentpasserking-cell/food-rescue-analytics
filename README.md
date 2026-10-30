# Food Rescue Analytics: AI Based Surplus Food Prediction System

Prototype for the ES-451 Minor Project (Group CSE_025). It reads donor, donation, NGO and claim data,
cleans it, forecasts daily surplus food (kg) and estimates which donations are at risk of being wasted,
and shows everything in a Streamlit dashboard.

## Setup (Python 3.10+)
```
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python generate_sample_data.py   # creates SIMULATED data in sample_data/
python run_pipeline.py           # optional: runs everything in the terminal
streamlit run app.py             # opens the dashboard in the browser
```

##  run
`python run_pipeline.py path/to/your_file.xlsx`.

Only the **Donations** table is required, with a date column and a quantity column. Other columns
(food category, donor type, prepared/expiry time, claims, NGOs) make the results better.
If your column names differ, add them to `ALIASES` in `config.py`. If your data uses other words
for wasted food, add them to `WASTE_WORDS`.

**Wastage rule:** a donation counts as wasted if its status
says so, its delivery failed or expired, or it was never claimed.

## Files and project modules
| File | Synopsis module |
|---|---|
| `data_loader.py` | 1. Data collection and cleaning |
| `app.py` (Overview tab) | 2. Exploratory analysis and dashboard |
| `forecasting.py` (`make_features`) | 3. Feature engineering |
| `forecasting.py` | 4. Surplus forecasting (baselines, Random Forest, Gradient Boosting) |
| `risk_model.py` | 5. Wastage-risk classification |
| `app.py` | 6. Web interface and reports (CSV downloads) |
| `run_pipeline.py` | 8. Evaluation (MAE, RMSE, MAPE, precision, recall, F1, ROC-AUC) |

## Important notes
* The sample data is **simulated**. 
* Models are tested on the most recent 20% of days, never on shuffled data (time-based split).
* A model is only worth reporting if it beats the baselines; the app tells you when it does not.
* Real donation data is usually noisy; forecast errors (especially MAPE) can be large for small categories.
* The risk model uses only information known at donation time (no claim details), to avoid data leakage.
* Ideas to extend: add XGBoost or Prophet, festival/holiday features, weather, location-wise forecasts, a database.
