#command prompt commands to run the dashboard
#cd code_path
#streamlit run app.py  or  python -m streamlit run app.py


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================
# STREAMLIT APP CONFIG
# ============================
st.set_page_config(page_title="UK Stock Market Analysis", layout="wide")
st.title("📊 UK Stock Market Analysis - MSc Project")

# Sidebar navigation
section = st.sidebar.radio("Navigate", ["Load Data", "EDA", "Feature Engineering", "Modeling", "Results"])

# ============================
# 1. Load Dataset
# ============================
if section == "Load Data":
    st.header("1. Load Dataset")
    try:
        df = pd.read_csv("UK Stock Market 1988-2024.csv")
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        st.dataframe(df.head())
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Place 'UK Stock Market 1988-2024.csv' in the same folder.")
        
# ==============================

# ----------------------------
# Load data at the top
# ----------------------------
df = pd.read_csv("UK Stock Market 1988-2024.csv")

# Identify OHLC columns
ohlc = {}
for col in df.columns:
    c = col.lower()
    if "open" in c: ohlc['open'] = col
    if "high" in c: ohlc['high'] = col
    if "low" in c: ohlc['low'] = col
    if "close" in c: ohlc['close'] = col

# ============================
# 2. Exploratory Data Analysis
# ============================
if section == "EDA":
    st.header("2. Exploratory Data Analysis")
    df = pd.read_csv("UK Stock Market 1988-2024.csv")
    
    st.subheader("Dataset Overview")
    st.write(df.describe())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    # Time series of closing price
    close_col = [c for c in df.columns if 'close' in c.lower()]
    if close_col:
        st.subheader("Closing Price Time Series")
        plt.figure(figsize=(12, 5))
        plt.plot(df[close_col[0]], color='blue')
        plt.title(f"{close_col[0]} Time Series")
        plt.xlabel("Time")
        plt.ylabel("Price")
        st.pyplot(plt)

    # Distribution histogram
    st.subheader("Close Price Distribution")
    if close_col:
        plt.figure(figsize=(8, 5))
        plt.hist(df[close_col[0]].dropna(), bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of {close_col[0]}")
        st.pyplot(plt)

    # Boxplot for outliers
    st.subheader("Boxplot of Closing Price")
    if close_col:
        plt.figure(figsize=(8, 5))
        sns.boxplot(y=df[close_col[0]], color='lightgreen')
        plt.title(f"Boxplot of {close_col[0]}")
        st.pyplot(plt)

    # Pair plot for numeric features
    st.subheader("Pairplot of Numeric Features")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        sample_df = df[numeric_cols].sample(min(1000, len(df)))  # avoid too large pairplot
        sns.pairplot(sample_df)
        st.pyplot(plt)
        
# ----------------------------
# Feature Engineering Section
# ----------------------------
if section == "Feature Engineering":
    st.header("3. Feature Engineering")

    if 'close' in ohlc:
        close_col = ohlc['close']

        # Sidebar sliders for moving average windows
        window1 = st.sidebar.slider("Short-term MA window", 5, 50, 20)
        window2 = st.sidebar.slider("Long-term MA window", 50, 200, 100)

        # Compute moving averages
        df[f'MA_{window1}'] = df[close_col].rolling(window=window1).mean()
        df[f'MA_{window2}'] = df[close_col].rolling(window=window2).mean()

        st.subheader(f"Showing last 10 rows with moving averages ({window1}, {window2})")
        st.dataframe(df[[close_col, f'MA_{window1}', f'MA_{window2}']].tail(10))

        # Plot closing price with moving averages
        st.subheader("Closing Price with Moving Averages")
        plt.figure(figsize=(12, 5))
        plt.plot(df[close_col], label="Close", color='blue')
        plt.plot(df[f'MA_{window1}'], label=f"MA {window1}", color='orange')
        plt.plot(df[f'MA_{window2}'], label=f"MA {window2}", color='green')
        plt.title("Closing Price and Moving Averages")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

    st.success("Feature Engineering Completed ✅")

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


# ============================
# GLOBAL MODEL TRAINING (accessible everywhere)
# ============================
if 'close' in ohlc:
    close_col = ohlc['close']

    # Compute MA features if missing
    ma_cols = [col for col in df.columns if "MA_" in col]
    if len(ma_cols) == 0:
        df['MA_20'] = df[close_col].rolling(window=20).mean()
        df['MA_100'] = df[close_col].rolling(window=100).mean()
        ma_cols = ['MA_20', 'MA_100']

    # Drop NaNs
    X = df[ma_cols].dropna()
    y = df[close_col].iloc[X.index]

    # Downsample for speed
    sample_size = min(len(X), 5000)
    X_sample = X.tail(sample_size)
    y_sample = y.tail(sample_size)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, shuffle=False
    )

    # ---- Linear Regression ----
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    # ---- SVR ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svr_model = SVR(kernel='linear', C=1.0, epsilon=0.05)
    svr_model.fit(X_train_scaled, y_train)
    svr_pred = svr_model.predict(X_test_scaled)
    svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
    svr_mae = mean_absolute_error(y_test, svr_pred)
    svr_r2 = r2_score(y_test, svr_pred)

    # ---- Random Forest ----
    rf_model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    # ---- Metrics DataFrame ----
    metrics_df = pd.DataFrame({
        "Model": ["Linear Regression", "SVR", "Random Forest"],
        "RMSE": [lr_rmse, svr_rmse, rf_rmse],
        "MAE": [lr_mae, svr_mae, rf_mae],
        "R²": [lr_r2, svr_r2, rf_r2]
    })

# ----------------------------
# Modeling Section
# ----------------------------
if section == "Modeling":
    st.header("4. Modeling and Result Evaluation")

    if 'close' not in ohlc:
        st.error("Close column not found. Cannot run modeling.")
    else:
        # Display metrics
        st.subheader("Linear Regression")
        st.write(f"RMSE: {lr_rmse:.2f}, MAE: {lr_mae:.2f}, R²: {lr_r2:.3f}")

        st.subheader("Support Vector Regression (SVR)")
        st.write(f"RMSE: {svr_rmse:.2f}, MAE: {svr_mae:.2f}, R²: {svr_r2:.3f}")

        st.subheader("Random Forest Regressor")
        st.write(f"RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.3f}")

        # Plot predictions
        st.subheader("Actual vs Predicted Closing Price")
        plt.figure(figsize=(12, 5))
        plt.plot(y_test.values, label="Actual", color='blue')
        plt.plot(lr_pred, label="Linear Regression", color='red')
        plt.plot(svr_pred, label="SVR", color='purple')
        plt.plot(rf_pred, label="Random Forest", color='green')
        plt.legend()
        st.pyplot(plt)

        # Metrics table
        st.subheader("Evaluation Metrics Table")
        st.dataframe(metrics_df)

        # Comparison plots
        st.subheader("Graphical Comparison")
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].bar(metrics_df["Model"], metrics_df["RMSE"], color=['red', 'purple', 'green']); ax[0].set_title("RMSE")
        ax[1].bar(metrics_df["Model"], metrics_df["MAE"], color=['red', 'purple', 'green']); ax[1].set_title("MAE")
        ax[2].bar(metrics_df["Model"], metrics_df["R²"], color=['red', 'purple', 'green']); ax[2].set_title("R²")
        st.pyplot(fig)

        # Recommendation
        best_model_index = np.argmin([lr_mae, svr_mae, rf_mae])
        best_model_name = metrics_df["Model"].iloc[best_model_index]
        st.subheader("Recommendation")
        st.info(f"Best model (lowest RMSE): **{best_model_name}**")
        
# ============================
# Results Section
# ============================
if section == "Results":
    st.header("5. Final Results & Forecasting")

    if 'close' not in ohlc:
        st.error("Close column not found. Cannot run results.")
    else:
        close_col = ohlc['close']

        # Ensure Date column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            date_col = 'Date'
        else:
            df['Date'] = pd.RangeIndex(start=0, stop=len(df))  # fallback
            date_col = 'Date'

        # -------------------
        # Actual vs Predicted Plot
        # -------------------
        st.subheader("Actual vs Predicted Closing Price (Test Set)")
        plt.figure(figsize=(14, 6))
        plt.plot(df[date_col].iloc[X_test.index], y_test, label="Actual", color='blue')
        plt.plot(df[date_col].iloc[X_test.index], lr_pred, label="Linear Regression", color='red')
        plt.plot(df[date_col].iloc[X_test.index], svr_pred, label="SVR", color='purple')
        plt.plot(df[date_col].iloc[X_test.index], rf_pred, label="Random Forest", color='green')
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title("Actual vs Predicted Closing Price")
        plt.legend()
        st.pyplot(plt)

        # -------------------
        # Future Forecasting
        # -------------------
        st.subheader("Future Forecasting")

        horizon = st.radio("Select Forecast Horizon:", ["3 Months", "6 Months", "9 Months", "12 Months"])
        months_map = {"3 Months": 90, "6 Months": 180, "9 Months": 270, "12 Months": 365}
        future_days = months_map[horizon]

        # Start from last known values
        last_known_date = df[date_col].iloc[-1]
        last_close = df[close_col].iloc[-1]

        # Use SVR(best performing usually) for forecasting
        forecast_model = svr_model
        forecast_dates = pd.date_range(start=last_known_date, periods=future_days+1, freq="D")[1:]
        forecast_values = []

        # Get last rolling values
        df_temp = df.copy()
        for future_date in forecast_dates:
            # Compute features
            ma_20 = df_temp[close_col].rolling(20).mean().iloc[-1]
            ma_100 = df_temp[close_col].rolling(100).mean().iloc[-1]
            features = np.array([[ma_20, ma_100]])
            
            # Scale features (important for SVR!)
            features_scaled = scaler.transform(features)

            # Predict next closing price-
            pred_price = forecast_model.predict(features)[0]
            forecast_values.append(pred_price)

            # Append prediction to temp df for next iteration
            new_row = pd.DataFrame({date_col: [future_date], close_col: [pred_price]})
            df_temp = pd.concat([df_temp, new_row], ignore_index=True)

        # -------------------
        # Plot Future Forecast
        # -------------------
        st.subheader(f"Forecasted Closing Price for {horizon}")
        plt.figure(figsize=(14, 6))
        plt.plot(df[date_col], df[close_col], label="Historical", color="blue")
        plt.plot(forecast_dates, forecast_values, label=f"Forecast ({horizon})", color="orange")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title(f"Stock Price Forecast ({horizon})")
        plt.legend()
        st.pyplot(plt)

        st.success(f"Forecasting completed for {horizon} ✅")








