# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Environmental Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Generate realistic environmental data
# ---------------------------
def generate_realistic_environmental_data(days=3):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq="1H")
    
    data = []
    for i, t in enumerate(timestamps):
        hour = t.hour
        day_of_week = t.weekday()
        
        # Temperature pattern - cooler at night, warmer during day
        base_temp = 20 + 2 * np.sin(2 * np.pi * i / 24)  # Daily cycle
        day_variation = 3 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly cycle
        temp = base_temp + day_variation + np.random.normal(0, 0.7)
        
        # Humidity pattern - higher at night, lower during day
        base_humidity = 60 - 15 * np.sin(2 * np.pi * (i-2) / 24)  # Offset by 2 hours
        weather_effect = 10 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly pattern
        hum = base_humidity + weather_effect + np.random.normal(0, 2)
        hum = max(30, min(hum, 90))
        
        # AQI pattern - higher during day, lower at night
        base_aqi = 30 + 25 * np.sin(2 * np.pi * (i+4) / 24)  # Offset by 4 hours
        
        # Add some random pollution events (10% chance each hour)
        aqi = base_aqi + np.random.normal(0, 5)
        if np.random.random() < 0.1:
            pollution_event = np.random.uniform(20, 50)
            aqi += pollution_event  # single-hour spike (approx)
        
        # Weekend effect (lower AQI on weekends)
        if day_of_week >= 5:
            aqi -= 10
        
        aqi = max(0, min(aqi, 200))
        
        data.append({
            "timestamp": t,
            "temperature": round(temp, 1),
            "humidity": round(hum, 1),
            "air_quality": round(aqi, 1)
        })
    
    df = pd.DataFrame(data)
    return df

# ---------------------------
# AQI Category function
# ---------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢", "#00E400"
    elif aqi <= 100:
        return "Moderate", "🟡", "#FFFF00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#FF7E00"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#FF0000"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#8F3F97"
    else:
        return "Hazardous", "⚫", "#7E0023"

# ---------------------------
# AQI Prediction function using Prophet
# ---------------------------
def predict_aqi_prophet(df, hours=24, interval_width=0.80):
    """
    Train Prophet on historical hourly AQI and forecast next `hours` hours.
    Returns dataframe with columns: timestamp, yhat, yhat_lower, yhat_upper
    If model fails, returns empty DataFrame.
    """
    try:
        prophet_df = df[['timestamp', 'air_quality']].copy()
        prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'air_quality': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

        # Need some minimum points
        if prophet_df.shape[0] < 10:
            return pd.DataFrame([])

        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=interval_width,
            changepoint_prior_scale=0.05
        )

        # Fit
        m.fit(prophet_df)

        # Make future dataframe for hourly steps
        future = m.make_future_dataframe(periods=hours, freq='H')

        # Predict
        forecast = m.predict(future)

        # Take only the forecasted future rows
        last_hist = prophet_df['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_hist].copy()
        out = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'ds': 'timestamp', 'yhat': 'yhat', 'yhat_lower': 'yhat_lower', 'yhat_upper': 'yhat_upper'}
        )
        out['yhat'] = out['yhat'].round(1)
        out['yhat_lower'] = out['yhat_lower'].round(1)
        out['yhat_upper'] = out['yhat_upper'].round(1)
        return out.reset_index(drop=True)
    except Exception as e:
        # Don't crash the app; return empty and log to console
        print("Prophet predict error:", e)
        return pd.DataFrame([])

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("🌡️ Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data Explorer", "Predictions", "About"])

# Initialize data in session_state
if 'environmental_data' not in st.session_state:
    st.session_state.environmental_data = generate_realistic_environmental_data(3)
    st.session_state.last_refresh = datetime.now()

# Auto-refresh every hour (append new hourly datapoint)
current_time = datetime.now()
if (current_time - st.session_state.last_refresh) >= timedelta(hours=1):
    latest = st.session_state.environmental_data.iloc[-1]
    new_time = current_time.replace(minute=0, second=0, microsecond=0)
    hour = new_time.hour
    temp = 20 + 5 * np.sin((hour - 6) / 12 * np.pi) + np.random.normal(0, 0.7)
    hum = 60 - 10 * np.sin((hour - 6) / 12 * np.pi) + np.random.normal(0, 2)
    aqi = 30 + 25 * np.sin((hour + 4) / 12 * np.pi) + np.random.normal(0, 5)
    new_data = pd.DataFrame([{
        "timestamp": new_time,
        "temperature": round(temp, 1),
        "humidity": round(hum, 1),
        "air_quality": round(aqi, 1)
    }])
    st.session_state.environmental_data = pd.concat([st.session_state.environmental_data, new_data], ignore_index=True)
    if len(st.session_state.environmental_data) > 72:  # keep ~3 days
        st.session_state.environmental_data = st.session_state.environmental_data.iloc[1:].reset_index(drop=True)
    st.session_state.last_refresh = current_time

# Use the data
df = st.session_state.environmental_data.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------------------------
# Pages
# ---------------------------
if page == "Dashboard":
    latest_temp = df.iloc[-1]["temperature"]
    latest_hum = df.iloc[-1]["humidity"]
    latest_aq = df.iloc[-1]["air_quality"]
    aq_category, aq_emoji, aq_color = get_aqi_category(latest_aq)

    st.title("🌡️ Environmental Dashboard")
    st.markdown("### Real-time monitoring of temperature, humidity, and air quality")
    next_refresh = st.session_state.last_refresh + timedelta(hours=1)
    st.info(f"Data refreshes automatically every hour. Next refresh: {next_refresh.strftime('%H:%M:%S')}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Temperature", value=f"{latest_temp}°C",
                  delta=f"{latest_temp - df.iloc[-2]['temperature']:.1f}°C" if len(df) > 1 else None)
    with col2:
        st.metric(label="Humidity", value=f"{latest_hum}%",
                  delta=f"{latest_hum - df.iloc[-2]['humidity']:.1f}%" if len(df) > 1 else None)
    with col3:
        st.metric(label="Air Quality", value=f"{latest_aq} AQI")
    with col4:
        st.metric(label="AQI Status", value=f"{aq_emoji} {aq_category}")

    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Statistics", "🌡️ AQI Analysis"])

    with tab1:
        # Create separate figures instead of one with multiple y-axes
        st.subheader("Temperature")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["temperature"],
            name="Temperature (°C)",
            mode="lines+markers",
            line=dict(color='red', width=2)
        ))
        fig_temp.update_layout(
            xaxis=dict(title="Time", rangeslider=dict(visible=True), type="date"),
            yaxis=dict(title="Temperature (°C)"),
            height=300
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
        st.subheader("Humidity")
        fig_hum = go.Figure()
        fig_hum.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["humidity"],
            name="Humidity (%)",
            mode="lines+markers",
            line=dict(color='blue', width=2)
        ))
        fig_hum.update_layout(
            xaxis=dict(title="Time", rangeslider=dict(visible=True), type="date"),
            yaxis=dict(title="Humidity (%)"),
            height=300
        )
        st.plotly_chart(fig_hum, use_container_width=True)
        
        st.subheader("Air Quality")
        fig_aqi = go.Figure()
        fig_aqi.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["air_quality"],
            name="Air Quality (AQI)",
            mode="lines+markers",
            line=dict(color='green', width=2)
        ))
        fig_aqi.update_layout(
            xaxis=dict(title="Time", rangeslider=dict(visible=True), type="date"),
            yaxis=dict(title="Air Quality (AQI)"),
            height=300
        )
        st.plotly_chart(fig_aqi, use_container_width=True)

    with tab2:
        st.subheader("Data Statistics")
        stats_df = pd.DataFrame({
            "Metric": ["Temperature (°C)", "Humidity (%)", "Air Quality (AQI)"],
            "Average": [df["temperature"].mean(), df["humidity"].mean(), df["air_quality"].mean()],
            "Min": [df["temperature"].min(), df["humidity"].min(), df["air_quality"].min()],
            "Max": [df["temperature"].max(), df["humidity"].max(), df["air_quality"].max()],
            "Std Dev": [df["temperature"].std(), df["humidity"].std(), df["air_quality"].std()]
        })
        st.dataframe(stats_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Temperature Distribution")
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Histogram(x=df["temperature"], nbinsx=20, name="Temperature"))
            fig_temp.update_layout(title="Temperature Distribution", xaxis_title="Temperature (°C)", yaxis_title="Count")
            st.plotly_chart(fig_temp, use_container_width=True)
        with c2:
            st.subheader("Humidity Distribution")
            fig_hum = go.Figure()
            fig_hum.add_trace(go.Histogram(x=df["humidity"], nbinsx=20, name="Humidity"))
            fig_hum.update_layout(title="Humidity Distribution", xaxis_title="Humidity (%)", yaxis_title="Count")
            st.plotly_chart(fig_hum, use_container_width=True)

    with tab3:
        st.subheader("Air Quality Analysis")
        df["aqi_category"] = df["air_quality"].apply(lambda x: get_aqi_category(x)[0])
        df["aqi_color"] = df["air_quality"].apply(lambda x: get_aqi_category(x)[2])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Average AQI by Hour of Day")
            df["hour"] = df["timestamp"].dt.hour
            hourly_avg = df.groupby("hour")["air_quality"].mean().reset_index()
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=hourly_avg["hour"], y=hourly_avg["air_quality"],
                                     marker_color=[get_aqi_category(aq)[2] for aq in hourly_avg["air_quality"]]))
            fig_bar.update_layout(xaxis_title="Hour of Day", yaxis_title="Average AQI")
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            st.markdown("#### AQI Category Distribution")
            category_counts = df["aqi_category"].value_counts()
            pie_colors = [get_aqi_category(df[df["aqi_category"] == cat]["air_quality"].mean())[2] for cat in category_counts.index]
            fig_pie = go.Figure(data=[go.Pie(labels=category_counts.index, values=category_counts.values, hole=.3, marker=dict(colors=pie_colors))])
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### AQI Timeline with Categories")
        fig_timeline = go.Figure()
        for category in df["aqi_category"].unique():
            cat_data = df[df["aqi_category"] == category]
            color = get_aqi_category(cat_data["air_quality"].mean())[2]
            fig_timeline.add_trace(go.Scatter(
                x=cat_data["timestamp"],
                y=cat_data["air_quality"],
                name=category,
                mode="markers",
                marker=dict(color=color, size=8),
                hovertemplate="Time: %{x}<br>AQI: %{y}<br>Category: " + category
            ))
        fig_timeline.update_layout(xaxis_title="Time", yaxis_title="AQI", height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)

elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.subheader("Raw Environmental Data")
    st.dataframe(df, use_container_width=True)
    st.subheader("Data Summary")
    st.write(f"Dataset contains {len(df)} records from {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Temperature Records", len(df))
    with c2:
        st.metric("Total Humidity Records", len(df))
    with c3:
        st.metric("Total AQI Records", len(df))
    st.download_button(label="Download Data as CSV", data=df.to_csv(index=False), file_name=f"environmental_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

elif page == "Predictions":
    st.title("🔮 AQI Predictions")
    st.markdown("### 24-Hour Air Quality Index Forecast")

    with st.spinner("Generating AQI predictions..."):
        predictions = predict_aqi_prophet(df, hours=24, interval_width=0.80)

    if predictions.empty:
        st.warning("Could not generate predictions (not enough data or model error).")
    else:
        # Summary metrics
        avg_aqi = predictions["yhat"].mean()
        min_aqi = predictions["yhat"].min()
        max_aqi = predictions["yhat"].max()
        cat, emoji, color = get_aqi_category(avg_aqi)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Average Predicted AQI", f"{avg_aqi:.1f}")
        with c2:
            st.metric("Minimum Predicted AQI", f"{min_aqi:.1f}")
        with c3:
            st.metric("Maximum Predicted AQI", f"{max_aqi:.1f}")
        with c4:
            st.metric("Predicted Category", f"{emoji} {cat}")

        # Plot: historical + forecast + uncertainty band
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["air_quality"], name="Historical AQI", mode="lines+markers", line=dict(color="blue", width=2)))
        fig.add_trace(go.Scatter(x=predictions["timestamp"], y=predictions["yhat"], name="Predicted AQI (median)", mode="lines+markers", line=dict(color="red", width=3)))
        # uncertainty band
        fig.add_trace(go.Scatter(
            x=pd.concat([predictions["timestamp"], predictions["timestamp"][::-1]]),
            y=pd.concat([predictions["yhat_upper"], predictions["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval"
        ))
        fig.update_layout(title="24-Hour AQI Forecast", xaxis_title="Time", yaxis_title="AQI", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        display_df = predictions[["timestamp", "yhat"]].copy()
        display_df = display_df.rename(columns={"yhat": "Predicted AQI"})
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        # add category + emoji
        display_df["Category"] = predictions["yhat"].apply(lambda x: get_aqi_category(x)[0])
        display_df["Status"] = predictions["yhat"].apply(lambda x: get_aqi_category(x)[1])
        st.subheader("Detailed Predictions")
        st.dataframe(display_df.set_index("timestamp"), use_container_width=True)
        st.warning("Note: Predictions are estimates based on simulated historical patterns and Prophet model.")

elif page == "About":
    st.title("ℹ️ About This Dashboard")
    st.markdown("""
    ## Environmental Monitoring Dashboard
    
    This dashboard provides real-time monitoring and historical analysis of environmental metrics:
    - Temperature (°C)
    - Humidity (%)
    - Air Quality Index (AQI)
    
    Predictions use Prophet with daily & weekly seasonality.
    """)
    st.table(pd.DataFrame([
        {"AQI Range": "0-50", "Category": "Good", "Emoji": "🟢"},
        {"AQI Range": "51-100", "Category": "Moderate", "Emoji": "🟡"},
        {"AQI Range": "101-150", "Category": "Unhealthy for Sensitive Groups", "Emoji": "🟠"},
        {"AQI Range": "151-200", "Category": "Unhealthy", "Emoji": "🔴"},
        {"AQI Range": "201-300", "Category": "Very Unhealthy", "Emoji": "🟣"},
        {"AQI Range": "301+", "Category": "Hazardous", "Emoji": "⚫"}
    ]))

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Environmental Dashboard v1.0 | © 2023")

if page != "About":
    st.markdown("---")
    st.caption(f"Data updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Next refresh at: {(st.session_state.last_refresh + timedelta(hours=1)).strftime('%H:%M:%S')}")
