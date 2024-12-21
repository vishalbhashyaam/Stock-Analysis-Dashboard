# 📊 Stock Analysis Dashboard

An interactive Streamlit dashboard for visualizing and analyzing stock price data using regression models. This dashboard mimics Tableau's storytelling approach to provide insightful visualizations, regression analysis, and future price predictions.

---

## 🚀 Features

- **Upload and Analyze**: Upload CSV files containing stock data with dates and prices.
- **Visualize Stock Prices**: View price trends over time using interactive line charts.
- **Price Distribution**: Analyze price distributions with histograms.
- **Regression Analysis**: Apply various regression models (Linear, Ridge, Lasso, Random Forest, SVR) to predict stock prices.
- **Future Predictions**: Input future dates to predict stock prices based on historical data.
- **Tableau-like Storytelling**: Intuitive layout with a focus on story-driven insights.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Stock-Analysis-Dashboard.git
cd Stock-Analysis-Dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📂 File Structure

```
/stock-analysis-dashboard
│
├── app.py                     # Main Streamlit application file
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (this file)
```

---

## 📋 Requirements

Add the following dependencies to your `requirements.txt`:

```text
streamlit
pandas
plotly
scikit-learn
```

---

## 📊 How to Use

1. **Upload CSV File**: Upload a CSV file with at least two columns – 'Date' and stock price.
2. **Visualize Trends**: Explore stock price trends and distribution.
3. **Select Regression Model**: Choose a regression model from the sidebar.
4. **Predict Future Prices**: Input a future date to predict stock prices.

---

## 📈 Example CSV Format

```csv
Date,Stock_Price
2023-01-01,150
2023-01-02,152
2023-01-03,148
2023-01-04,155
```

---

## 🔍 Key Sections

- **Price Over Time**: Interactive line chart to view historical trends.
- **Price Distribution**: Histogram showing price frequency distribution.
- **Regression Analysis**: Apply models to fit data and visualize predicted vs. actual prices.
- **Prediction**: Enter a future date to predict stock prices.

---

## 🧑‍💻 Code Overview (app.py)

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Streamlit Page Config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📊 Interactive Stock Analysis Dashboard")

# Sidebar for File Upload
if 'assets' not in st.session_state:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_file is not None:
        st.session_state.assets = pd.read_csv(uploaded_file)

if 'assets' in st.session_state:
    assets = st.session_state.assets
    assets['Date'] = pd.to_datetime(assets['Date'], dayfirst=True, errors='coerce')
    assets['Date_Ordinal'] = assets['Date'].apply(lambda x: x.toordinal() - assets['Date'].min().toordinal())

    st.subheader("📅 Price Over Time")
    fig = px.line(assets, x='Date', y=assets.columns[1], title="Stock Price Movement")
    st.plotly_chart(fig)

    st.subheader("📊 Price Distribution")
    hist_fig = px.histogram(assets, x=assets.columns[1], nbins=30)
    st.plotly_chart(hist_fig)

    # Regression and Prediction Section
    st.sidebar.subheader("Regression Analysis")
    model_option = st.sidebar.selectbox("Select Model", ["Linear", "Ridge", "Lasso", "Random Forest", "SVR"])
    future_date = st.sidebar.date_input("Select Future Date")

    # Model Training
    X = assets[['Date_Ordinal']]
    y = assets[assets.columns[1]]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    st.subheader("📈 Predicted vs Actual Prices")
    reg_fig = px.line(assets, x='Date', y=assets.columns[1])
    reg_fig.add_scatter(x=assets['Date'], y=predictions, mode='lines')
    st.plotly_chart(reg_fig)

    if future_date:
        ordinal_date = pd.to_datetime(future_date).toordinal() - assets['Date'].min().toordinal()
        future_prediction = model.predict([[ordinal_date]])[0]
        st.sidebar.write(f"Predicted Price on {future_date}: ${future_prediction:.2f}")
```

---

## 📜 License

MIT License

---

## 🤝 Contributing

Feel free to fork this repository and create pull requests. Contributions are welcome!

---

## 📧 Contact

For questions or suggestions, feel free to reach out to me at [vishalbhashyaam@gmail.com](mailto:vishalbhashyaam@gmail.com).
