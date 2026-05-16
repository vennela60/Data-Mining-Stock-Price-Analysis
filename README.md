Project Title:
Data Mining Techniques with Feature Selection for Predicting Stock Prices on UK Stock Exchange Data

This repository contains the full implementation of a machine‑learning pipeline designed to forecast UK stock prices using data mining, feature engineering, and feature selection techniques.
The project is based on the dissertation “Insertion of Data Mining Techniques with Feature Selection to Predict Share Price on Stock Exchange Data” by Vennela Yaganti.

Project Overview:
This project builds a complete predictive system that:
* Cleans and preprocesses UK stock market data
* Performs Exploratory Data Analysis (EDA)
* Applies multiple feature selection techniques
* Trains ML models (Linear Regression, SVR, Random Forest)
* Evaluates performance using RMSE, MAE, R²
* Deploys the best model (SVR) in an interactive Streamlit dashboard
The system is designed to help analysts and investors make data‑driven decisions using historical stock data.

📂 Dataset:
* Source: Kaggle — UK Daily Historical Stock Market (1988–2024)
* Contains:• Date, Ticker, Company Name
* Open, High, Low, Close, Adj Close
* Volume
* Size: 4.4M+ rows (downsampled for computation)
“The study uses a broad data collection of UK stock market information, tracking 1,026 companies from 1988 to 2024.”

Data Preprocessing:
The preprocessing pipeline includes:
* Handling missing values (forward/backward fill, mode)
* Removing duplicates
* Removing negative or illogical price values
* Normalizing numerical features
* Converting date fields
* Outlier detection using boxplots
“Errors, missing pieces of data, and varying levels of data in the raw market data can hurt how your model works.”

Exploratory Data Analysis (EDA):
EDA includes:
* Summary statistics
* Price distribution plots
* Correlation heatmaps
* Time‑series visualizations
* Outlier detection
* Ticker frequency analysis
“EDA has shown significant structures in the trends in stocks and statistical testing.”

Feature Engineering
Created technical indicators:
* Daily returns
* Price range
* Volatility
* Moving averages (MA5, MA10, MA20, MA100)
* Momentum
* Volume ratios
* Date‑based features (Month, Year, DayOfWeek)

Feature Selection Techniques:
Three methods were applied:
✔️ Correlation‑based filtering
✔️ Statistical tests (F‑test)
✔️ Recursive Feature Elimination (RFE)

Final selected features included:
* Price_Change_2d
* Daily_Return
* HL_Spread
* Volume_Ratio
* Price_VS_MA5
* Month, Year, DayOfWeek, DayOfMonth
* OC_Change
* Price_Range
“The three‑pronged approach… allowed reducing the dimensionality of the dataset without the loss of vital information.”

Machine Learning Models:
The following models were trained:
* Linear Regression
* Support Vector Regression (SVR)
* Random Forest Regression

Performance Summary:
Model	RMSE	MAE	R²	Notes	
SVR	161.42	40.56	-0.019	Best overall	
Linear Regression	183.68	106.32	-0.32	Weak linear fit	
Random Forest	231.49	100.36	-1.09	Overfitting	
“SVR performed the best across the board… with the lowest RMSE and MAE values.”

Dashboard Deployment (Streamlit):
The project includes a full interactive dashboard with:
* Data upload
* EDA visualizations
* Feature engineering
* Model training
* Forecasting (3, 6, 9, 12 months)
* Actual vs Predicted plots
“The implementation of Random Forest into an interactive dashboard constitutes practical value when used by investors and analysts.”

Project Structure:
├── data/
│   └── uk_stock_data.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── models.py
│   └── evaluation.py
├── dashboard/
│   └── app.py
├── README.md
└── requirements.txt


How to Run the Project:
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Run the Streamlit dashboard
streamlit run dashboard/app.py


📈 Key Findings
* Feature selection significantly improves model performance
* SVR is the most stable and accurate model
* Ensemble models overfit on downsampled data
* Moving averages and volatility indicators are strong predictors
* Deployment enables real‑time decision support
