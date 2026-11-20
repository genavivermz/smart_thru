import numpy as np
import pandas as pd
import plotly.express as px
from routing_engine import calculate_order_release
import streamlit as st

st.set_page_config(
    page_title="SmartThru | AI Order Routing & Unit Economics", layout="wide"
)

st.title("☕ SmartThru: Predictive Drive-Thru Order Timing & Throughput Engine")
st.markdown(
    "Synchronizing live traffic ETAs with XGBoost prep-time forecasting to"
    " eliminate drive-thru bottlenecks."
)

# Create two tabs: One for Apple AI/ML appeal, one for TMT Investment Banking appeal
tab1, tab2 = st.tabs([
    "Customer & ML Routing Simulator",
    "Store Unit Economics & Throughput",
])

# Tab 1: Customer & ML routing simulator
with tab1:
  col1, col2 = st.columns([1, 1.3])

  with col1:
    st.subheader("1. Live Driver & Store Telemetry")
    distance_km = st.slider(
        "Distance from Store (km)",
        min_value=0.1,
        max_value=10.0,
        value=4.0,
        step=0.1,
    )
    traffic_multiplier = st.selectbox(
        "Current Traffic Condition",
        [
            "Light (1.0x)",
            "Moderate (1.3x)",
            "Heavy Rush Hour (1.8x)",
            "Gridlock (2.4x)",
        ],
        index=2,
    )

    mult_map = {
        "Light (1.0x)": 1.0,
        "Moderate (1.3x)": 1.3,
        "Heavy Rush Hour (1.8x)": 1.8,
        "Gridlock (2.4x)": 2.4,
    }
    base_drive_sec = (distance_km / 40.0) * 3600  # Assume 40 km/h city average
    traffic_eta_sec = int(base_drive_sec * mult_map.get(traffic_multiplier))

    st.subheader("2. Order & Store Queue Inputs")
    item_count = st.slider("Items in Order", 1, 6, 2)
    complexity_per_item = st.slider(
        "Avg Drink Complexity (1=Black Coffee, 4=Frappuccino)", 1.0, 4.0, 2.5
    )
    queue_depth = st.slider("Current Cars/Orders Ahead in Queue", 0, 18, 8)
    baristas = st.slider("Baristas on Shift", 2, 5, 4)
    hour = st.slider("Hour of Day (24h)", 6, 20, 8)
    manual_override = st.checkbox("User Manual Override (Force Send ASAP)")

  with col2:
    st.subheader("3. Real-Time Engine Decision")

    # Simulate coordinates based on slider distance (~0.009 deg latitude per km)
    store_coords = (34.0562, -117.1825)
    driver_coords = (34.0562 + (distance_km * 0.009), -117.1825)

    order_features = {
        "item_count": item_count,
        "total_complexity": round(item_count * complexity_per_item, 1),
        "hour_of_day": hour,
        "is_rush_hour": 1 if hour in (7, 8, 9, 12, 13) else 0,
        "queue_depth": queue_depth,
        "baristas_on_shift": baristas,
    }

    decision = calculate_order_release(
        driver_coords=driver_coords,
        store_coords=store_coords,
        traffic_eta_seconds=traffic_eta_sec,
        order_features=order_features,
        manual_override=manual_override,
    )

    # Display key metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Traffic ETA", f"{round(traffic_eta_sec / 60, 1)} mins")
    m2.metric(
        "XGBoost Prep Forecast",
        f"{round(decision.get('predicted_prep_sec') / 60, 1)} mins",
    )
    m3.metric("Recommended Hold", f"{decision.get('hold_seconds')} sec")

    if "ASAP" in decision.get("status"):
      st.success(f"### Decision: {decision.get('status')}")
    else:
      st.warning(f"### Decision: {decision.get('status')}")

    st.info(
        f"**Trigger Logic:** {decision.get('reason')} | **GPS Distance:**"
        f" {decision.get('distance_meters')}m"
    )

    # Visual timeline breakdown
    timeline_df = pd.DataFrame({
        "Phase": [
            "Virtual Hold Window",
            "Barista Prep Time",
            "Freshness Buffer",
        ],
        "Seconds": [
            decision.get("hold_seconds"),
            decision.get("predicted_prep_sec"),
            45,
        ],
    })
    fig = px.bar(
        timeline_df,
        x="Seconds",
        y="Phase",
        orientation="h",
        title="Synchronized Arrival & Preparation Timeline",
        color="Phase",
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Store unit economics
with tab2:
  st.subheader("Peak-Hour Throughput & Margin Impact Simulator (1,000 Vehicles)")
  st.markdown(
      "Models the financial impact of eliminating premature order arrivals and"
      " customer line abandonment (reneging)."
  )

  f_col1, f_col2 = st.columns([1, 2])

  with f_col1:
    avg_ticket = st.number_input("Average Order Ticket ($)", value=8.50, step=0.50)
    peak_hours_per_day = st.slider("Peak Rush Hours per Day", 2, 6, 4)
    gross_margin_pct = st.slider("Store Incremental Gross Margin (%)", 50, 85, 72)

  with f_col2:
    # Deterministic simulation comparing Traditional ASAP vs. SmartThru Routing
    np.random.seed(42)
    n_cars = 1000
    base_service_time = np.random.normal(
        95, 20, n_cars
    )  # Seconds at pickup window

    # Traditional ASAP: Early arrivals clog the lane & block ready orders (+55s avg bottleneck penalty)
    asap_bottleneck = np.random.exponential(55, n_cars)
    asap_total_wait = base_service_time + asap_bottleneck

    # SmartThru: Synchronized arrivals cut bottleneck delays by ~68%
    smart_bottleneck = asap_bottleneck * 0.32
    smart_total_wait = base_service_time + smart_bottleneck

    # Throughput (cars/hr) & Reneging (customers leaving if wait > 4.5 mins)
    asap_throughput = round(3600 / np.mean(asap_total_wait), 1)
    smart_throughput = round(3600 / np.mean(smart_total_wait), 1)
    throughput_lift_pct = round(
        ((smart_throughput - asap_throughput) / asap_throughput) * 100, 1
    )

    asap_renege_rate = round(np.mean(asap_total_wait > 240) * 100, 1)
    smart_renege_rate = round(np.mean(smart_total_wait > 240) * 100, 1)

    # Annualized Store Revenue Lift (365 days)
    extra_cars_per_day = (
        smart_throughput - asap_throughput
    ) * peak_hours_per_day
    annual_rev_lift = extra_cars_per_day * avg_ticket * 365
    annual_ebitda_lift = annual_rev_lift * (gross_margin_pct / 100.0)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Peak Throughput Lift",
        f"+{throughput_lift_pct}%",
        f"{asap_throughput} → {smart_throughput} cars/hr",
    )
    c2.metric(
        "Customer Reneging Drop",
        f"-{round(asap_renege_rate - smart_renege_rate, 1)}%",
        f"{asap_renege_rate}% → {smart_renege_rate}%",
    )
    c3.metric(
        "Est. Annual Revenue Lift / Store",
        f"${annual_rev_lift:,.0f}",
        f"+${annual_ebitda_lift:,.0f} Gross Profit",
    )

    # Comparison Chart
    comp_df = pd.DataFrame({
        "System": ["Traditional ASAP", "SmartThru AI Routing"],
        "Avg Window Wait (Sec)": [
            round(np.mean(asap_total_wait), 1),
            round(np.mean(smart_total_wait), 1),
        ],
        "Peak Cars / Hour": [asap_throughput, smart_throughput],
    })
    fig2 = px.bar(
        comp_df,
        x="System",
        y="Avg Window Wait (Sec)",
        color="System",
        title="Average Drive-Thru Window Wait Time Comparison",
    )
    st.plotly_chart(fig2, use_container_width=True)