# ✧ The SmartThru

**predictive drive-thru order routing, syncing traffic ETAs with ML prep-time forecasts.**

* → static "ASAP" mobile ordering prevents drive-thrus from operating at full efficiency.
* → early arrivals clog the pickup window, drinks sit and melt, and long lines make drivers leave.
* **⚡︎ SmartThru** holds mobile tickets in a virtual queue and fires them at the exact moment needed, your drink finishing right as you pull up.

### how it works

* 1. **prep-time forecasting (`XGBoost`):** trained on 5,000 synthetic rush hour orders with non-linear queue bottlenecks and natural noise. predicts barista prep time ($T_{\text{prep}}$) within **±15.7s MAE (0.943 R²)** using item count, drink complexity, rush hour flags, queue depth, and active baristas.
* 2. **spatial geofencing (`Haversine`):** tracks distance to the store and triggers an instant **ASAP** release inside a **200m radius,**avoiding constant GPS polling.
* 3. **dynamic hold engine:** calculates optimal ticket delay ($D$) with a manual override toggle:
   $$D = \max(0, T_{\text{travel}} - T_{\text{prep}} - T_{\text{buffer}})$$


### simulation highlights *(1,000 peak hour vehicles)*

* **+33.1% peak throughput:** `23.9 → 31.8 cars/hr` capacity lift
* **-7.6% line abandonment:** `7.6% → 0.0% customer reneging drop
* **+98,039 est. annual revenue / store:** `+$70,588` incremental gross profit
* **±15.7s prep forecast MAE:** `0.943 R²` across non-linear rush hour queues

### dual-view dashboard (`Streamlit` + `Plotly`)

* **ML & routing view:** tweak driver distance, traffic multipliers, drink complexity, and queue depth to watch the XGBoost model and geofence logic adjust ticket hold windows live.
* **store unit economics view:** model store-level revenue lift, throughput velocity, and margin sensitivity across ticket sizes and rush hour windows.

### la stack

`Python` `XGBoost` `Scikit-Learn` `Pandas` `NumPy` `Streamlit` `Plotly`

### quickstart

1. **clone the repository** (`genavivermz/smart_thru`):
   ```bash
   gh repo clone genavivermz/smart_thru
   cd smart_thru
   ```

2. **set up environment & install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **train the model & launch the dashboard:**
   ```bash
   python train_model.py
   streamlit run app.py
   ```
