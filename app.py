import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Stock Regression Dashboard", layout="wide", initial_sidebar_state="expanded")

# Title and Description
st.title("📈 Stock Analysis and Regression Dashboard")
st.markdown("""
Welcome to the Stock Analysis and Regression Dashboard. Upload your dataset to explore stock prices, perform regression analysis, and visualize predictions.
""")

# Sidebar Styling
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/684/684908.png", width=80)
st.sidebar.header("⚙️ Control Panel")

# File Uploader with Session State
if 'assets' not in st.session_state:
    uploaded_file = st.sidebar.file_uploader("📤 Upload CSV Data", type=["csv"])
    if uploaded_file is not None:
        st.session_state.assets = pd.read_csv(uploaded_file)

if 'assets' in st.session_state:
    assets = st.session_state.assets

    # Data Preprocessing (Only apply .str before datetime conversion)
    if assets['Date'].dtype == 'object':
        assets['Date'] = assets['Date'].str.replace('/', '-')
        assets['Date'] = assets['Date'].str.replace(r'-(\d)-', r'-0\1-', regex=True)
        assets['Date'] = pd.to_datetime(assets['Date'], dayfirst=True, errors='coerce')

    # Create ordinal date column
    assets['Date_Ordinal'] = assets['Date'].apply(lambda x: x.toordinal() - assets['Date'].min().toordinal())

    # Show Data
    st.subheader("🗃️ Raw Data")
    with st.expander("View Raw Data"):
        st.write(assets.head())

    # Stock Selection
    stock_options = [col for col in assets.columns if 'Price' in col]
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

    if model_option == "Linear Regression":
        model = LinearRegression()
    elif model_option == "Ridge Regression":
        model = Ridge(alpha=alpha)
    elif model_option == "Lasso Regression":
        model = Lasso(alpha=alpha)
    elif model_option == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    elif model_option == "Support Vector Regressor":
        model = SVR(kernel=kernel)

    model.fit(X, y)
    full_predictions = model.predict(X)

    # Regression Plot with Interactive Visualization
    st.subheader("📈 Regression Analysis")
    regression_fig = px.line(assets, x='Date', y=selected_stock, title="Actual vs Predicted Prices")
    if model_option == "Random Forest Regressor":
        regression_fig.add_scatter(x=assets['Date'], y=full_predictions, mode='markers', marker=dict(color='orange'), name=f"Predicted ({model_option})")
        regression_fig.add_scatter(x=assets['Date'], y=full_predictions, mode='lines', line=dict(color='cyan'), name=f"Prediction Line ({model_option})")
    else:
        regression_fig.add_scatter(x=assets['Date'], y=full_predictions, mode='lines', name=f"Predicted ({model_option})")
    st.plotly_chart(regression_fig, use_container_width=True)

    # Prediction Section
    st.sidebar.subheader("📅 Make Predictions")
    future_date = st.sidebar.date_input("Select Future Date")
    if future_date:
        ordinal_date = pd.to_datetime(future_date).toordinal() - assets['Date'].min().toordinal()
        future_prediction = model.predict([[ordinal_date]])[0]
        st.sidebar.write(f"Predicted {selected_stock} Price on {future_date}: **${future_prediction:.2f}**")
