import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Setting seed for consistent results on each run
np.random.seed(42)
n_samples = 5000

# Generating realistic drive-thru features
item_choices = list(range(1, 7))
item_count = np.random.choice(
    item_choices, size=n_samples, p=[0.45, 0.28, 0.15, 0.07, 0.03, 0.02]
)
avg_item_complexity = np.random.uniform(1.0, 4.0, size=n_samples)
total_complexity = np.round(item_count * avg_item_complexity, 1)

# Time of day & rush hour
hour_of_day = np.random.randint(6, 21, size=n_samples)
rush_hours = list((7, 8, 9, 12, 13))
is_rush_hour = np.isin(hour_of_day, rush_hours).astype(int)

# Queue depth (cars/orders ahead) is higher during rush hour
queue_depth = np.where(
    is_rush_hour == 1,
    np.random.poisson(lam=7, size=n_samples),
    np.random.poisson(lam=2, size=n_samples),
)
queue_depth = np.clip(queue_depth, 0, 18)

# Baristas on shift (peaks at rush, falls at slow hours)
rush_staff = list((3, 4, 5))
slow_staff = list((2, 3))
baristas_on_shift = np.where(
    is_rush_hour == 1,
    np.random.choice(rush_staff, size=n_samples, p=[0.2, 0.5, 0.3]),
    np.random.choice(slow_staff, size=n_samples, p=[0.6, 0.4]),
)

# Calculate prep time in seconds
base_time = 35
complexity_time = total_complexity * 18
bottleneck_delay = (queue_depth**1.25) * (45 / baristas_on_shift)

# Real-world variance (normal noise, occasional operational delays)
normal_noise = np.random.normal(0, 12, size=n_samples)
hiccup_options = list((0, 45, 90))
random_hiccups = np.random.choice(
    hiccup_options, size=n_samples, p=[0.90, 0.07, 0.03]
)

prep_time_seconds = (
    base_time
    + complexity_time
    + bottleneck_delay
    + normal_noise
    + random_hiccups
)
prep_time_seconds = np.clip(prep_time_seconds, 30, 900)

# Save data to CSV
df = pd.DataFrame({
    'item_count': item_count,
    'total_complexity': total_complexity,
    'hour_of_day': hour_of_day,
    'is_rush_hour': is_rush_hour,
    'queue_depth': queue_depth,
    'baristas_on_shift': baristas_on_shift,
    'prep_time_seconds': np.round(prep_time_seconds, 1),
})
df.to_csv('synthetic_drivethru_orders.csv', index=False)

# Split data into Training (80%) and Testing (20%) sets
X = df.drop('prep_time_seconds', axis=1)
y = df['prep_time_seconds']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Machine Learning Model
model = xgb.XGBRegressor(
    n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
)
model.fit(X_train, y_train)

# Test accuracy and save trained model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Dataset saved: {len(df)} orders')
print(f'Model Mean Absolute Error (MAE): ±{mae:.1f} seconds')
print(f'Model R² Score: {r2:.3f}')

joblib.dump(model, 'prep_time_xgb_model.pkl')
print('Model saved as prep_time_xgb_model.pkl')
