import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
 
from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator
 
# ------------------------------------------------
# Page Configuration & Custom CSS
# ------------------------------------------------
 
st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)
 
# Custom CSS for a polished look
st.markdown("""
<style>

div[data-testid="stMetric"]{
    background-color:white;
    padding:20px;
    border-radius:15px;
    border:2px solid #D6EAF8;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

div[data-testid="stMetric"] label{
    color:#2C3E50 !important;
    font-weight:bold;
}

div[data-testid="stMetric"] div{
    color:#0077B6 !important;
}

</style>
""", unsafe_allow_html=True)
 
# ------------------------------------------------
# Sidebar & Logic
# ------------------------------------------------
 
with st.sidebar:
    st.image("assets/OIP.52-e1770295315617.jpg", width=100)
    st.title("Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.info("Adjust the slider to change the prediction window for the RNN model.")
 
# ------------------------------------------------
# Data & Header
# ------------------------------------------------
 
loader = DataLoader("data/airline-passengers.csv")
df = loader.load_data()
 
st.title("✈️ Airline Passenger Analysis & Forecasting")
st.caption("Predicting global travel trends using Recurrent Neural Networks (RNN)")
 
# ------------------------------------------------
# Metrics & Overview Tabs
# ------------------------------------------------
 
tab1, tab2 = st.tabs(["🚀 Model Performance", "🔎 Exploratory Data Analysis"])
 
with tab1:
    st.subheader("Model Accuracy Metrics")
    mae, mse, rmse = Evaluator().evaluate()
   
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", delta_color="inverse")
    m2.metric("Mean Squared Error (MSE)", f"{mse:.2f}", delta_color="inverse")
    m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}", delta_color="inverse")
 
with tab2:
    col_a, col_b = st.columns([1, 2])
   
    with col_a:
        st.subheader("Raw Data")
        st.dataframe(df, height=350)
   
    with col_b:
        st.subheader("Historical Trend")
        fig = px.line(df, x=df.index, y="Passengers",
                      template="plotly_white",
                      color_discrete_sequence=['#007bff'])
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
 
# ------------------------------------------------
# Forecasting Section
# ------------------------------------------------
 
st.markdown("---")
st.header("🔮 Generate Future Forecast")
 
if st.button("Run RNN Model"):
    with st.spinner("Analyzing temporal patterns..."):
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
 
    st.success(f"Successfully generated forecast for {future_months} months!")
 
    # Layout for Results
    res_col1, res_col2 = st.columns([1, 2])
 
    with res_col1:
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True)
       
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )
 
    with res_col2:
        st.subheader("Combined Projection")
       
        # Create a combined chart with Plotly
        fig_combined = go.Figure()
       
        # Historical Data
        fig_combined.add_trace(go.Scatter(
            x=df.index, y=df["Passengers"],
            name="Historical", line=dict(color="#6c757d", width=2)
        ))
       
        # Forecasted Data
        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted Passengers"],
            name="Forecast", line=dict(color="#ff7f0e", width=3, dash='dot')
        ))
       
        fig_combined.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_combined, use_container_width=True)
