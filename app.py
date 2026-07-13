import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Airline Passenger Forecaster",
    
    layout="wide"
)

# ------------------------------------------------
# Custom CSS
# ------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.main {
    background-color: #0e1117;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #1f2937;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #374151;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

/* Metric Label */
[data-testid="stMetricLabel"] {
    color: #d1d5db !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Metric Value */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 34px !important;
    font-weight: bold !important;
}

/* Metric Delta */
[data-testid="stMetricDelta"] {
    color: #22c55e !important;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    width: 100%;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    future_months = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=24,
        value=12
    )

    st.info(
        "Adjust the slider to choose how many months to forecast."
    )

# ------------------------------------------------
# Load Data
# ------------------------------------------------

loader = DataLoader("data/airline_passengers.csv")
df = loader.load_data()

# ------------------------------------------------
# Header
# ------------------------------------------------

col1, col2 = st.columns([1, 5])

with col1:
    st.image("assets/52-e1770295315617.jpg", width=120)

with col2:
    st.title("✈️ Airline Passenger Analysis & Forecasting")
    st.caption("Predicting future airline passenger demand using RNN, LSTM and GRU models.")
# ------------------------------------------------
# Tabs
# ------------------------------------------------

tab1, tab2 = st.tabs(
    ["🚀 Model Performance", "📊 Exploratory Data Analysis"]
)

# ------------------------------------------------
# Model Metrics
# ------------------------------------------------

with tab1:

    st.subheader("Model Accuracy")

    mae, mse, rmse = Evaluator().evaluate()

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")

# ------------------------------------------------
# Data Analysis
# ------------------------------------------------

with tab2:

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Dataset")
        st.dataframe(df, height=350)

    with right:
        st.subheader("Historical Passenger Trend")

        fig = px.line(
            df,
            x=df.index,
            y="Passengers",
            template="plotly_white"
        )

        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Forecast Section
# ------------------------------------------------

st.markdown("---")
st.header("🔮 Generate Future Forecast")

if st.button("Run RNN Model"):

    with st.spinner("Generating forecast..."):

        forecaster = Forecaster()
        future = forecaster.forecast(future_months)

        last_date = df.index[-1]

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })

    st.success(f"Forecast generated for the next {future_months} months.")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.subheader("Forecast Values")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        csv = forecast_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    with col2:

        st.subheader("Forecast Visualization")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Passengers"],
                name="Historical",
                line=dict(width=2)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["Predicted Passengers"],
                name="Forecast",
                line=dict(width=3, dash="dot")
            )
        )

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)