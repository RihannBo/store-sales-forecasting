# 📊 Store Sales Forecasting

## Project Overview
This project focuses on forecasting daily sales for multiple stores and product families using time series modeling techniques.

The goal is to predict future sales based on historical patterns, seasonality, promotions, and holidays.

---

## Dataset
The project is based on the **Store Sales - Time Series Forecasting** dataset (Kaggle), which includes:

- Daily sales data per store and product family
- Promotion information
- Holiday events
- Store metadata

---

## Exploratory Data Analysis (EDA)

Key insights from the data:

-  Sales show a clear upward trend over time  
-  Strong weekly and monthly seasonality patterns  
-  Promotions significantly increase sales  
-  Holidays impact demand with noticeable spikes or drops  
-  Sales distribution is highly skewed (many small values, few large peaks)  

---

##  Feature Engineering

The following features were created to improve model performance:

### Time-based Features
- Year, month, day
- Day of week, day of year
- Trend index (time progression)

### Lag Features
- `lag7`, `lag14`, `lag21`, `lag28`

### Rolling Statistics
- Rolling mean: `rmean7`, `rmean14`, `rmean28`
- Rolling std (volatility): `rstd7`, `rstd14`, `rstd28`

### Seasonality Encoding
- Cyclical features:
  - `month_sin`, `month_cos`
  - `dow_sin`, `dow_cos`

### External Features
- Promotion indicator (`onpromotion`, `is_promo`)
- Holiday indicator (`is_holiday`)

---

##  Model

Model used:

- `HistGradientBoostingRegressor` (Scikit-learn)

Why:
- Handles large datasets efficiently
- Supports non-linear relationships
- Robust to missing values

---

##  Evaluation

Metric used:
- **RMSLE (Root Mean Squared Logarithmic Error)**

Result:

```text
Train RMSLE: ~0.41  
Validation RMSLE: ~0.41