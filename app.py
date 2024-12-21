import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

# Page Config
st.set_page_config(page_title="Stock Regression Dashboard", layout="wide", initial_sidebar_state="expanded")

# Title and Description
st.title("📈 Stock Analysis and Regression Dashboard")
st.markdown("""
Welcome to the Stock Analysis and Regression Dashboard. Analyze stock prices, perform regression, and visualize predictions.
""")

# Sidebar Styling
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/684/684908.png", width=80)
st.sidebar.header("⚙️ Control Panel")

# Load Data Directly from Data Folder
DATA_PATH = "Data/stock_prices.csv"

if os.path.exists(DATA_PATH):
    assets = pd.read_csv(DATA_PATH)

    # Data Preprocessing (Date)
    if assets['Date'].dtype == 'object':
        assets['Date'] = assets['Date'].str.replace('/', '-')
        assets['Date'] = assets['Date'].str.replace(r'-(\d)-', r'-0\1-', regex=True)
        assets['Date'] = pd.to_datetime(assets['Date'], dayfirst=True, errors='coerce')

    # Handle missing dates
    assets = assets.dropna(subset=['Date'])

    # ---- PRICE CLEANING FIX ----
    stock_options = [col for col in assets.columns if 'Price' in col]
    for col in stock_options:
        assets[col] = assets[col].replace({',': ''}, regex=True)  # Remove commas
        assets[col] = pd.to_numeric(assets[col], errors='coerce')  # Convert to float
    assets = assets.dropna(subset=stock_options)  # Drop rows with NaN prices
    # ----------------------------

    # Create ordinal date column
    assets['Date_Ordinal'] = assets['Date'].apply(lambda x: x.toordinal() - assets['Date'].min().toordinal())

    # Show Data
    st.subheader("🗃️ Raw Data")
    with st.expander("View Raw Data"):
        st.write(assets.head())

    # Stock Selection
    selected_stock = st.sidebar.selectbox("📊 Select Stock", stock_options)

    # Visualization
    st.subheader(f"📅 {selected_stock} Price Over Time")
    fig = px.line(assets, x='Date', y=selected_stock, title=f"{selected_stock} Price Over Time")
    fig.update_layout(hovermode='x unified', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot
    st.subheader("📊 Scatter Plot of Price vs Date")
    scatter_fig = px.scatter(assets, x='Date', y=selected_stock, title=f"{selected_stock} Price Distribution Over Time")
    scatter_fig.update_traces(marker=dict(size=6, opacity=0.6))
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Histogram
    st.subheader("📈 Price Distribution")
    hist_fig = px.histogram(assets, x=selected_stock, nbins=30, title=f"{selected_stock} Price Distribution")
    hist_fig.update_layout(xaxis_title=f"{selected_stock} Price", bargap=0.2)
    st.plotly_chart(hist_fig, use_container_width=True)

    # Regression Model Selection
    st.sidebar.subheader("🔍 Select Regression Model")
    model_option = st.sidebar.selectbox("Model", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest Regressor", "Support Vector Regressor"])
    alpha = st.sidebar.slider("🔧 Regularization (alpha)", 0.01, 10.0, 1.0)
    n_estimators = st.sidebar.slider("🌲 Number of Trees (Random Forest)", 10, 300, 100)
    kernel = st.sidebar.selectbox("🎛 Kernel (SVR)", ["linear", "rbf", "poly"])

    # Regression
    X = assets[['Date_Ordinal']]
    y = assets[selected_stock]

    # Check for low variance
    if assets[selected_stock].nunique() < 5:
        st.warning(f"{selected_stock} has too little variance for effective regression.")
    else:
        # Train-test split to avoid overfitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        try:
            # Model selection
            if model_option == "Linear Regression":
                model = LinearRegression()
            elif model_option == "Ridge Regression":
                model = Ridge(alpha=alpha)
            elif model_option == "Lasso Regression":
                model = Lasso(alpha=alpha)
            elif model_option == "Random Forest Regressor":
                model = RandomForestRegressor(n_estimators=min(n_estimators, len(X) // 2), random_state=42)
            elif model_option == "Support Vector Regressor":
                model = SVR(kernel=kernel)

            # Model fitting
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Evaluate model
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            st.sidebar.write(f"📊 Model Performance (Test Set):")
            st.sidebar.write(f"Mean Squared Error: **{mse:.2f}**")
            st.sidebar.write(f"R² Score: **{r2:.2f}**")

            # Full predictions for visualization
            full_predictions = model.predict(X)

            # Regression plot
            st.subheader("📈 Regression Analysis")
            regression_fig = px.line(assets, x='Date', y=selected_stock, title="Actual vs Predicted Prices")
            if model_option == "Random Forest Regressor":
                regression_fig.add_scatter(
                    x=assets['Date'],
                    y=full_predictions,
                    mode='markers+lines',
                    marker=dict(color='orange', size=8),
                    line=dict(color='red'),
                    name=f"Predicted ({model_option})"
                )
            else:
                regression_fig.add_scatter(
                    x=assets['Date'],
                    y=full_predictions,
                    mode='lines',
                    name=f"Predicted ({model_option})"
                )
            st.plotly_chart(regression_fig, use_container_width=True)

        except Exception as e:
            st.error(f"🚨 Model Training Error: {str(e)}")

    # Prediction Section (Fix for Feature Name Warning)
    st.sidebar.subheader("📅 Make Predictions")
    future_date = st.sidebar.date_input("Select Future Date")
    if future_date:
        ordinal_date = pd.to_datetime(future_date).toordinal() - assets['Date'].min().toordinal()
        ordinal_date = pd.DataFrame([[ordinal_date]], columns=['Date_Ordinal'])
        future_prediction = model.predict(ordinal_date)[0]
        st.sidebar.write(f"Predicted {selected_stock} Price on {future_date}: **${future_prediction:.2f}**")
else:
    st.error("🚨 Data file not found in 'Data/' folder. Please add a CSV file named 'stock_prices.csv'.")
