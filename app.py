import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from core.preprocessor import LoadPreprocessor
from core.model import ForecastModel
from core.decision_support import DecisionSupport
import os

# Page Config
st.set_page_config(page_title="Powercast-AI Dashboard", layout="wide")

st.title("⚡ Powercast-AI: Load Forecasting & Decision Support")
st.markdown("---")

# Sidebar - User Inputs
st.sidebar.header("🔧 Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Historical Load CSV", type=["csv"])
look_back = st.sidebar.slider("Look-back Window (Hours)", 6, 48, 24)
horizon = st.sidebar.slider("Forecast Horizon (Hours)", 1, 72, 24)

st.sidebar.header("🏭 Generator Capacities (MW)")
unit1 = st.sidebar.number_input("Unit 1", value=300)
unit2 = st.sidebar.number_input("Unit 2", value=250)
unit3 = st.sidebar.number_input("Unit 3", value=200)
capacities = [unit1, unit2, unit3]

# Initialize modules
preprocessor = LoadPreprocessor()
model = ForecastModel(look_back=look_back)
decision_support = DecisionSupport(capacities)

# Main Logic
if uploaded_file is not None or os.path.exists('data/historical_load.csv'):
    # Load Data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('data/historical_load.csv')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    st.subheader("📊 Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Raw vs. Smoothed Load")
        df['smoothed_load'] = preprocessor.smooth_data(df['load'].values)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['load'], name="Actual Load", line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smoothed_load'], name="Smoothed Load", line=dict(color='blue', width=2)))
        st.plotly_chart(fig, use_container_width=True)

    # Prepare Data for Training
    data_values = df['smoothed_load'].values.reshape(-1, 1)
    normalized_data = preprocessor.normalize_data(data_values)
    X, y = preprocessor.prepare_sliding_window(normalized_data, look_back)
    
    # Train Model
    with st.spinner("Training forecasting model..."):
        model.train(X, y.flatten())
    
    # Forecasting
    last_window = normalized_data[-look_back:]
    predictions_norm = model.multi_step_forecast(last_window, horizon)
    predictions = preprocessor.inverse_transform(predictions_norm.reshape(-1, 1)).flatten()
    
    # Future timestamps
    last_ts = df['timestamp'].iloc[-1]
    future_ts = [last_ts + timedelta(hours=i+1) for i in range(horizon)]
    
    with col2:
        st.write("Future Load Forecast")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df['timestamp'].iloc[-48:], y=df['load'].iloc[-48:], name="Historical"))
        fig_pred.add_trace(go.Scatter(x=future_ts, y=predictions, name="Forecast", line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_pred, use_container_width=True)

    # Decision Support
    st.markdown("---")
    st.subheader("💡 Decision Support & Generation Planning")
    
    # Unit Commitment for Peak Load in horizon
    peak_load = np.max(predictions)
    avg_load = np.mean(predictions)
    
    on_units, off_units, total_cap = decision_support.recommend_units(peak_load)
    maintenance_indices, threshold = decision_support.identify_maintenance_windows(predictions)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Peak Predicted Load", f"{peak_load:.2f} MW")
    m_col2.metric("Total Online Capacity", f"{total_cap} MW")
    m_col3.metric("Maintenance Threshold", f"{threshold:.2f} MW")
    
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.success("✅ Suggested ON Units")
        for unit in on_units:
            st.write(f"- {unit}")
            
    with d_col2:
        st.warning("❌ Suggested OFF Units (Standby/Maintenance)")
        for unit in off_units:
            st.write(f"- {unit}")

    # Maintenance Planning
    st.subheader("🛠️ Maintenance Planning")
    maintenance_times = [future_ts[i] for i in maintenance_indices]
    if maintenance_times:
        st.info(f"Recommended maintenance windows identified during low-load periods (< {threshold:.2f} MW)")
        # Show first and last identified slots if continuous
        st.write(f"Key periods: {maintenance_times[0].strftime('%Y-%m-%d %H:%M')} to {maintenance_times[-1].strftime('%Y-%m-%d %H:%M')}")
    else:
        st.write("No specific maintenance windows identified for this horizon.")

else:
    st.info("Please upload a CSV file or ensure `data/historical_load.csv` exists to begin.")
    if st.button("Generate Sample Data"):
        import utils.generate_sample_data as gsd
        gsd.generate_sample_load_data()
        st.rerun()
