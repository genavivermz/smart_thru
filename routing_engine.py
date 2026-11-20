import math
import joblib
import pandas as pd

# Load trained XGBoost model
model = joblib.load('prep_time_xgb_model.pkl')

def haversine_distance_meters(lat1, lon1, lat2, lon2):
    """Calculates straight-line GPS distance in meters between driver and store."""
    R = 6371000 # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_order_release(
        driver_coords,
        store_coords,
        traffic_eta_seconds,
        order_features,
        manual_override=False,
):
    """Determines optimal order hold time using Geofencing + XGBoost Prep Prediction."""
    distance_m = haversine_distance_meters(
        driver_coords[0], driver_coords[1], store_coords[0], store_coords[1]
    )

    # Predict barista preparation time using trained XGBoost model
    features_df = pd.DataFrame([order_features])
    predicted_prep_sec = float(model.predict(features_df)[0])
    buffer_sec = 45 # Target drink finishing 45 seconds before car reaches window

    # Check override or 200m store geofence
    if manual_override or distance_m <= 200:
        return {
            'status': 'SEND ASAP',
            'hold_seconds': 0,
            'distance_meters': round(distance_m, 1),
            'predicted_prep_sec': round(predicted_prep_sec, 1),
            'reason': (
                'Manual Override Active'
                if manual_override
                else 'Inside 200m Store Geofence'
            ),
        }

    # Calculate optimal hold time: D = T_travel - T_prep - T_buffer
    hold_seconds = max(0, traffic_eta_seconds - predicted_prep_sec - buffer_sec)
    hold_mins = round(hold_seconds / 60, 1)

    if hold_seconds <= 30:
        status = 'SEND ASAP'
    elif hold_mins <= 3:
        status = f'HOLD 2-3 MINS ({int(hold_seconds)}s)'
    elif hold_mins <= 6:
        status = f'HOLD 4-6 MINS ({int(hold_seconds)}s)'
    else:
        status = f'HOLD {int(hold_mins)} MINS ({int(hold_seconds)}s)'

    return {
        'status': status,
        'hold_seconds': round(hold_seconds),
        'distance_meters': round(distance_m, 1),
        'predicted_prep_sec': round(predicted_prep_sec, 1),
        'reason': 'Synced with Traffic ETA & ML Prep Forecast',
  }

# Quick test run
if __name__ == '__main__':
    store = (34.0562, -117.1825) # Example Starbucks coordinates
    driver_far = (34.0850, -117.2100) # ~4 kilometers away in morning rush traffic

    sample_order = {
        'item_count': 2,
        'total_complexity': 5.5,
        'hour_of_day': 8,
        'is_rush_hour': 1,
        'queue_depth': 8,
        'baristas_on_shift': 4,
    }

    result = calculate_order_release(
        driver_coords=driver_far,
        store_coords=store,
        traffic_eta_seconds=540,  # 9 mins away in traffic
        order_features=sample_order,
    )

    print('Test Driver Result:', result)
