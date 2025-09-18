import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

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
    
    # Base patterns for more realistic data
    data = []
    for i, t in enumerate(timestamps):
        hour = t.hour
        day_of_week = t.weekday()
        
        # Temperature pattern - cooler at night, warmer during day
        # with some day-to-day variation
        base_temp = 20 + 2 * np.sin(2 * np.pi * i / 24)  # Daily cycle
        day_variation = 3 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly cycle
        temp = base_temp + day_variation + np.random.normal(0, 0.7)
        
        # Humidity pattern - higher at night, lower during day
        # inversely related to temperature but with some lag
        base_humidity = 60 - 15 * np.sin(2 * np.pi * (i-2) / 24)  # Offset by 2 hours
        weather_effect = 10 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly pattern
        hum = base_humidity + weather_effect + np.random.normal(0, 2)
        hum = max(30, min(hum, 90))  # Keep within reasonable bounds
        
        # AQI pattern - higher during day, lower at night
        # with some random pollution events
        base_aqi = 30 + 25 * np.sin(2 * np.pi * (i+4) / 24)  # Offset by 4 hours
        
        # Add some random pollution events (10% chance each hour)
        if np.random.random() < 0.1:
            pollution_event = np.random.uniform(20, 50)
            # Make events last for a few hours
            for j in range(min(4, len(timestamps) - i)):
                if i+j < len(data):
                    data[i+j]["air_quality"] += pollution_event * (1 - j/4)
        
        # Weekend effect (lower AQI on weekends)
        if day_of_week >= 5:  # Saturday or Sunday
            base_aqi -= 10
            
        aqi = base_aqi + np.random.normal(0, 5)
        aqi = max(0, min(aqi, 200))  # Cap at 200 for realism
        
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
def predict_aqi(df, hours=24):
    # Prepare data for Prophet
    prophet_df = df[['timestamp', 'air_quality']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=hours, freq='H')
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Filter only the future predictions
    future_forecast = forecast[forecast['ds'] > df['timestamp'].max()].copy()
    future_forecast['timestamp'] = future_forecast['ds']
    future_forecast['air_quality'] = future_forecast['yhat'].round(1)
    future_forecast['air_quality_lower'] = future_forecast['yhat_lower'].round(1)
    future_forecast['air_quality_upper'] = future_forecast['yhat_upper'].round(1)
    
    return future_forecast[['timestamp', 'air_quality', 'air_quality_lower', 'air_quality_upper']]

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("🌡️ Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data Explorer", "Predictions", "About"])

# Initialize session state for data if not exists
if 'environmental_data' not in st.session_state:
    st.session_state.environmental_data = generate_realistic_environmental_data(3)
    st.session_state.last_refresh = datetime.now()

# Check if it's time to refresh data (every hour)
current_time = datetime.now()
if (current_time - st.session_state.last_refresh) >= timedelta(hours=1):
    # Add a new data point while keeping the old data
    latest = st.session_state.environmental_data.iloc[-1]
    new_time = current_time.replace(minute=0, second=0, microsecond=0)
    
    # Generate new data point with similar patterns to previous data
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
    
    st.session_state.environmental_data = pd.concat([
        st.session_state.environmental_data,
        new_data
    ], ignore_index=True)
    
    # Remove the oldest data point if we have more than 3 days worth of data
    if len(st.session_state.environmental_data) > 72:  # 3 days * 24 hours
        st.session_state.environmental_data = st.session_state.environmental_data.iloc[1:]
    
    st.session_state.last_refresh = current_time

# Use the data from session state
df = st.session_state.environmental_data

if page == "Dashboard":
    # Get latest readings
    latest_temp = df.iloc[-1]["temperature"]
    latest_hum = df.iloc[-1]["humidity"]
    latest_aq = df.iloc[-1]["air_quality"]
    aq_category, aq_emoji, aq_color = get_aqi_category(latest_aq)
    
    # Main dashboard content
    st.title("🌡️ Environmental Dashboard")
    st.markdown("### Real-time monitoring of temperature, humidity, and air quality")
    
    # Auto-refresh info
    next_refresh = st.session_state.last_refresh + timedelta(hours=1)
    st.info(f"Data refreshes automatically every hour. Next refresh: {next_refresh.strftime('%H:%M:%S')}")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Temperature", 
            value=f"{latest_temp}°C",
            delta=f"{latest_temp - df.iloc[-2]['temperature']:.1f}°C" if len(df) > 1 else None
        )
    
    with col2:
        st.metric(
            label="Humidity", 
            value=f"{latest_hum}%",
            delta=f"{latest_hum - df.iloc[-2]['humidity']:.1f}%" if len(df) > 1 else None
        )
    
    with col3:
        st.metric(
            label="Air Quality", 
            value=f"{latest_aq} AQI"
        )
    
    with col4:
        st.metric(
            label="AQI Status", 
            value=f"{aq_emoji} {aq_category}"
        )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Statistics", "🌡️ AQI Analysis"])
    
    with tab1:
        # Main Plot
        fig = go.Figure()
    
        # Add traces with different y-axes
        fig.add_trace(go.Scatter(
            x=df["timestamp"], 
            y=df["temperature"], 
            name="Temperature (°C)", 
            mode="lines+markers",
            line=dict(color='red', width=2),
            yaxis="y"
        ))
    
        fig.add_trace(go.Scatter(
            x=df["timestamp"], 
            y=df["humidity"], 
            name="Humidity (%)", 
            mode="lines+markers",
            line=dict(color='blue', width=2),
            yaxis="y2"
        ))
    
        fig.add_trace(go.Scatter(
            x=df["timestamp"], 
            y=df["air_quality"], 
            name="Air Quality (AQI)", 
            mode="lines+markers",
            line=dict(color='green', width=2),
            yaxis="y3"
        ))
    
        # Update layout with multiple y-axes
        fig.update_layout(
            title="Environmental Metrics Over Time (Last 3 Days)",
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(
                title="Temperature (°C)",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                side="left"
            ),
            yaxis2=dict(
                title="Humidity (%)",
                titlefont=dict(color="blue"),
                tickfont=dict(color="blue"),
                anchor="free",
                overlaying="y",
                side="left"
            ),
            yaxis3=dict(
                title="Air Quality (AQI)",
                titlefont=dict(color="green"),
                tickfont=dict(color="green"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified",
            height=500
        )

    
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Data Statistics")
        
        # Calculate statistics
        stats_df = pd.DataFrame({
            "Metric": ["Temperature (°C)", "Humidity (%)", "Air Quality (AQI)"],
            "Average": [df["temperature"].mean(), df["humidity"].mean(), df["air_quality"].mean()],
            "Min": [df["temperature"].min(), df["humidity"].min(), df["air_quality"].min()],
            "Max": [df["temperature"].max(), df["humidity"].max(), df["air_quality"].max()],
            "Std Dev": [df["temperature"].std(), df["humidity"].std(), df["air_quality"].std()]
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Distribution")
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Histogram(x=df["temperature"], nbinsx=20, name="Temperature"))
            fig_temp.update_layout(title="Temperature Distribution", xaxis_title="Temperature (°C)", yaxis_title="Count")
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            st.subheader("Humidity Distribution")
            fig_hum = go.Figure()
            fig_hum.add_trace(go.Histogram(x=df["humidity"], nbinsx=20, name="Humidity"))
            fig_hum.update_layout(title="Humidity Distribution", xaxis_title="Humidity (%)", yaxis_title="Count")
            st.plotly_chart(fig_hum, use_container_width=True)
    
    with tab3:
        st.subheader("Air Quality Analysis")
        
        # Add AQI category to dataframe
        df["aqi_category"] = df["air_quality"].apply(lambda x: get_aqi_category(x)[0])
        df["aqi_color"] = df["air_quality"].apply(lambda x: get_aqi_category(x)[2])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Average AQI by Hour of Day")
            df["hour"] = df["timestamp"].dt.hour
            hourly_avg = df.groupby("hour")["air_quality"].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_avg["hour"], 
                y=hourly_avg["air_quality"],
                marker_color=[get_aqi_category(aq)[2] for aq in hourly_avg["air_quality"]]
            ))
            fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Average AQI")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### AQI Category Distribution")
            category_counts = df["aqi_category"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=.3,
                marker=dict(colors=[get_aqi_category(
                    df[df["aqi_category"] == cat]["air_quality"].mean()
                )[2] for cat in category_counts.index])
            )])
            st.plotly_chart(fig, use_container_width=True)
        
        # AQI timeline with colors
        st.markdown("#### AQI Timeline with Categories")
        fig = go.Figure()
        for category in df["aqi_category"].unique():
            cat_data = df[df["aqi_category"] == category]
            color = get_aqi_category(cat_data["air_quality"].mean())[2]
            fig.add_trace(go.Scatter(
                x=cat_data["timestamp"],
                y=cat_data["air_quality"],
                name=category,
                mode="markers",
                marker=dict(color=color, size=8),
                hovertemplate="Time: %{x}<br>AQI: %{y}<br>Category: " + category
            ))
        fig.update_layout(xaxis_title="Time", yaxis_title="AQI", height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    
    st.subheader("Raw Environmental Data")
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Data Summary")
    st.write(f"Dataset contains {len(df)} records from {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Temperature Records", len(df))
    with col2:
        st.metric("Total Humidity Records", len(df))
    with col3:
        st.metric("Total AQI Records", len(df))
    
    st.download_button(
        label="Download Data as CSV",
        data=df.to_csv(index=False),
        file_name=f"environmental_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

elif page == "Predictions":
    st.title("🔮 AQI Predictions")
    st.markdown("### 24-Hour Air Quality Index Forecast")
    
    # Generate predictions
    with st.spinner("Generating AQI predictions..."):
        predictions = predict_aqi(df, hours=24)
    
    # Display prediction summary
    st.subheader("Prediction Summary")
    
    # Calculate stats for predictions
    avg_aqi = predictions["air_quality"].mean()
    min_aqi = predictions["air_quality"].min()
    max_aqi = predictions["air_quality"].max()
    aq_category, aq_emoji, aq_color = get_aqi_category(avg_aqi)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Predicted AQI", f"{avg_aqi:.1f}")
    with col2:
        st.metric("Minimum Predicted AQI", f"{min_aqi:.1f}")
    with col3:
        st.metric("Maximum Predicted AQI", f"{max_aqi:.1f}")
    with col4:
        st.metric("Predicted Category", f"{aq_emoji} {aq_category}")
    
    # Plot predictions
    st.subheader("AQI Forecast with Confidence Interval")
    
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["air_quality"],
        name="Historical AQI",
        mode="lines+markers",
        line=dict(color="blue", width=2)
    ))
    
    # Add predicted data
    fig.add_trace(go.Scatter(
        x=predictions["timestamp"],
        y=predictions["air_quality"],
        name="Predicted AQI",
        mode="lines+markers",
        line=dict(color="red", width=3)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([predictions["timestamp"], predictions["timestamp"][::-1]]),
        y=pd.concat([predictions["air_quality_upper"], predictions["air_quality_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval"
    ))
    
    fig.update_layout(
        title="24-Hour AQI Forecast",
        xaxis_title="Time",
        yaxis_title="AQI",
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display prediction table
    st.subheader("Detailed Predictions")
    
    # Add category to predictions
    predictions["category"] = predictions["air_quality"].apply(lambda x: get_aqi_category(x)[0])
    predictions["emoji"] = predictions["air_quality"].apply(lambda x: get_aqi_category(x)[1])
    
    # Format for display
    display_df = predictions[["timestamp", "air_quality", "category", "emoji"]].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df.columns = ["Time", "Predicted AQI", "Category", "Status"]
    
    st.dataframe(display_df, use_container_width=True)
    
    # Warning about predictions
    st.warning("""
    **Note:** These predictions are based on historical patterns and should be considered as estimates only. 
    Actual AQI values may vary due to unforeseen environmental factors.
    """)

elif page == "About":
    st.title("ℹ️ About This Dashboard")
    
    st.markdown("""
    ## Environmental Monitoring Dashboard
    
    This dashboard provides real-time monitoring and historical analysis of environmental metrics including:
    - Temperature (°C)
    - Humidity (%)
    - Air Quality Index (AQI)
    
    ### Features
    - **Real-time metrics** with latest readings and trends
    - **Historical data visualization** with interactive charts
    - **Statistical analysis** of environmental data
    - **24-hour AQI predictions** using time series forecasting
    - **Data export** functionality for further analysis
    - **Automatic refresh** every hour with new data points
    
    ### Air Quality Index (AQI) Reference
    """)
    
    aqi_data = [
        {"AQI Range": "0-50", "Category": "Good", "Emoji": "🟢", "Color": "#00E400"},
        {"AQI Range": "51-100", "Category": "Moderate", "Emoji": "🟡", "Color": "#FFFF00"},
        {"AQI Range": "101-150", "Category": "Unhealthy for Sensitive Groups", "Emoji": "🟠", "Color": "#FF7E00"},
        {"AQI Range": "151-200", "Category": "Unhealthy", "Emoji": "🔴", "Color": "#FF0000"},
        {"AQI Range": "201-300", "Category": "Very Unhealthy", "Emoji": "🟣", "Color": "#8F3F97"},
        {"AQI Range": "301+", "Category": "Hazardous", "Emoji": "⚫", "Color": "#7E0023"}
    ]
    
    aqi_df = pd.DataFrame(aqi_data)
    st.table(aqi_df)
    
    st.markdown("""
    ### Prediction Model
    The AQI predictions are generated using Facebook's Prophet library, which is designed for forecasting 
    time series data with daily seasonality. The model considers:
    - Historical AQI patterns
    - Daily cycles (higher AQI during day, lower at night)
    - Weekly patterns (weekend vs weekday differences)
    
    ### Data Source
    The environmental data shown in this dashboard is generated using realistic simulation algorithms
    that mimic natural patterns and variations in temperature, humidity, and air quality.
    
    ### Technology Stack
    - Built with **Streamlit** for the web interface
    - **Plotly** for interactive visualizations
    - **Pandas** for data manipulation
    - **NumPy** for numerical computations
    - **Prophet** for time series forecasting
    """)
    
    st.info("""
    Note: This is a demonstration application using simulated data.
    In a real-world scenario, this would connect to actual environmental sensors.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Environmental Dashboard v1.0 | © 2023")

# Main footer
if page != "About":
    st.markdown("---")

    st.caption(f"Data updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Next refresh at: {(st.session_state.last_refresh + timedelta(hours=1)).strftime('%H:%M:%S')}")
