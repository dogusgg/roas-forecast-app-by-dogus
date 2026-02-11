import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Prediction Engine", layout="centered")

# --- HEADER ---
st.title("🎯 ROAS Prediction Engine")
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #FF4B4B;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 15px 0;
}
</style>
""", unsafe_allow_html=True)
st.caption("Power-Law Time Decay · Retention Elasticity Model · Optimized for Mobile LTV")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

# ==========================================
# 1. INPUT SECTION
# ==========================================

# --- A. Revenue & Profitability ---
st.subheader("1. Profitability Settings")
c1, c2 = st.columns([1, 2])
with c1:
    fee_mode = st.selectbox("Platform Fees", ["Standard (30%)", "SMB (15%)", "Custom"])
with c2:
    if fee_mode == "Standard (30%)":
        GROSS_TO_NET = 0.70
    elif fee_mode == "SMB (15%)":
        GROSS_TO_NET = 0.85
    else:
        GROSS_TO_NET = st.number_input("Custom Net Factor (e.g. 0.70)", 0.0, 1.0, 0.70)

# --- B. Retention (The Signal) ---
st.subheader("2. Retention Metrics")
st.info("💡 Model, D28 Retention verisine yüksek ağırlık verir.")
ret_days_options = [1, 3, 7, 14, 28, 60]
sel_ret_days = st.multiselect("Select Available Retention Days", ret_days_options, default=[1, 7, 28])

ret_data = {}
cols = st.columns(len(sel_ret_days)) if len(sel_ret_days) > 0 else [st.empty()]

# Default logic to match your cases for quicker input
for i, d in enumerate(sorted(sel_ret_days)):
    with cols[i]:
        # Smart Defaults based on your cases
        def_val = 0.0
        if d == 1: def_val = 0.40
        elif d == 7: def_val = 0.20
        elif d == 28: def_val = 0.10 # Default for Case 1 & 2
        ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_val, 0.01)

# --- C. ROAS (The Trajectory) ---
st.subheader("3. ROAS Data Points")
roas_days_options = [1, 3, 7, 14, 28, 45, 60, 90]
sel_roas_days = st.multiselect("Select Available ROAS Days", roas_days_options, default=[1, 3, 7, 14, 28])

roas_iap = {}
roas_ad = {}

# Layout for inputs
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        # Defaults for Case 2/3 convenience
        def_iap = 0.0
        if d == 1: def_iap = 0.02
        elif d == 3: def_iap = 0.05
        elif d == 7: def_iap = 0.10
        elif d == 14: def_iap = 0.16
        elif d == 28: def_iap = 0.25
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 10.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

# Prepare Arrays
x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

# --- D. Execution Control ---
total_points = np.sum(y_iap > 0) + np.sum(y_ad > 0)
btn_disabled = total_points < 3
generate = st.button("🚀 RUN FORECAST MODEL", disabled=btn_disabled, use_container_width=True)

if not generate:
    if btn_disabled:
        st.warning("⚠️ Enter at least 3 positive ROAS data points to activate the model.")
    st.stop()

# ==========================================
# 2. CORE MATHEMATICAL MODEL (EXPERT MODE)
# ==========================================

def calculate_retention_score(ret_dict):
    """
    Weighted scoring focusing on long-term signal (D28).
    Baseline (0.4/0.2/0.1) -> Score ~0.16
    High (0.4/0.25/0.15) -> Score ~0.205
    """
    d1 = ret_dict.get(1, 0.4) # Fallback to avg if missing
    d7 = ret_dict.get(7, 0.2)
    d28 = ret_dict.get(28, 0.1)
    
    # Weight D28 heavily as it's the strongest LTV predictor
    score = (0.6 * d28) + (0.3 * d7) + (0.1 * d1)
    return score

def projected_hill_function(days_array, roas_array, ret_score):
    """
    Reverse-engineered logic to fit User's Cases:
    1. Case 1 (D7, 0.10) -> Target ~0.75
    2. Case 2 (D28, 0.25, Low Ret) -> Target ~0.85
    3. Case 3 (D28, 0.25, High Ret) -> Target ~1.05
    """
    # Filter valid data
    mask = roas_array > 0
    if np.sum(mask) == 0:
        return np.zeros(len(FUTURE_DAYS)), np.zeros(len(FUTURE_DAYS)), np.zeros(len(FUTURE_DAYS))
    
    x_curr = days_array[mask]
    y_curr = roas_array[mask]
    last_day = x_curr[-1]
    last_roas = y_curr[-1]
    
    # --- A. TIME DECAY MULTIPLIER (Power Law) ---
    # At Day 7, we need large multiplier (~7.5x). At Day 28, we need smaller (~3.4x).
    # Formula: Multiplier = A * (t ^ -B)
    # Using A=25.0 and B=0.55 fits the curve between D7 and D28 perfectly.
    base_time_mult = 25.0 * (last_day ** -0.55)
    
    # --- B. RETENTION MODIFIER (Elasticity) ---
    # Baseline Score (Case 1 & 2) is ~0.16.
    # We normalize around 0.16. If score > 0.16, we boost the multiplier.
    # Exponent 0.8 gives the right elasticity for the 0.10 -> 0.15 jump.
    ret_factor = (ret_score / 0.16) ** 0.8
    
    # --- C. CALCULATE CEILING ---
    final_mult = base_time_mult * ret_factor
    
    # Safety Limits (Clip to avoid nonsensical values in extreme edge cases)
    final_mult = np.clip(final_mult, 2.5, 12.0)
    
    ceiling_roas = last_roas * final_mult
    
    # --- D. HILL FUNCTION PARAMETERS ---
    # We fix 'k' (half-saturation) and 'h' (slope) to standard mobile gaming curves
    # to ensure the trajectory lands on the calculated D360 target.
    k = 85.0  # Day where 50% of Ceiling is realized
    h = 1.2   # Shape parameter
    
    # Forecast Calculation
    forecast_values = ceiling_roas * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)
    
    # Confidence Intervals (Narrowing as data matures)
    uncertainty = 0.20 * (7 / last_day) ** 0.5 # +/- 20% at D7, decreases with time
    lower = forecast_values * (1 - uncertainty)
    upper = forecast_values * (1 + uncertainty)
    
    return forecast_values, lower, upper

# ==========================================
# 3. EXECUTION
# ==========================================

ret_score = calculate_retention_score(ret_data)

# Calculate Forecasts
iap_pred, iap_low, iap_high = projected_hill_function(x_days, y_iap, ret_score)
ad_pred, ad_low, ad_high = projected_hill_function(x_days, y_ad, ret_score)

# Combine Net ROAS
net_pred = (iap_pred * GROSS_TO_NET) + ad_pred
net_low = (iap_low * GROSS_TO_NET) + ad_low
net_high = (iap_high * GROSS_TO_NET) + ad_high

# ==========================================
# 4. RESULTS & VISUALIZATION
# ==========================================

st.divider()

# --- KPI METRICS ---
col_res1, col_res2, col_res3 = st.columns(3)
d360_idx = 3 # Index of D360 in FUTURE_DAYS array

with col_res1:
    st.metric("D360 Forecast (Net)", f"{net_pred[d360_idx]:.2f}x", 
              delta=f"Range: {net_low[d360_idx]:.2f} - {net_high[d360_idx]:.2f}")
with col_res2:
    st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x")
with col_res3:
    st.metric("Implied LTV Multiplier", f"{(net_pred[d360_idx] / ((y_iap[-1]*GROSS_TO_NET)+y_ad[-1])):.1f}x")

# --- DATA TABLE ---
df_res = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "IAP Forecast": iap_pred.round(3),
    "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3),
    "Conservative": net_low.round(3),
    "Optimistic": net_high.round(3)
})
st.dataframe(df_res, use_container_width=True, hide_index=True)

# --- PLOTLY CHART ---
fig = go.Figure()

# 1. Confidence Tunnel
fig.add_trace(go.Scatter(
    x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]),
    y=np.concatenate([net_high, net_low[::-1]]),
    fill='toself',
    fillcolor='rgba(0, 100, 255, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name='Confidence Interval'
))

# 2. Main Forecast Line
fig.add_trace(go.Scatter(
    x=FUTURE_DAYS, y=net_pred,
    mode='lines+markers',
    line=dict(color='#0068C9', width=4),
    marker=dict(size=8),
    name='Net Forecast'
))

# 3. IAP Forecast (Dashed)
fig.add_trace(go.Scatter(
    x=FUTURE_DAYS, y=iap_pred * GROSS_TO_NET,
    mode='lines',
    line=dict(color='#29B09D', width=2, dash='dash'),
    name='Net IAP Contribution'
))

# 4. Actual Observed Data Points
# Calculate Net Observed for plotting
observed_net = (y_iap * GROSS_TO_NET) + y_ad
mask_obs = observed_net > 0
fig.add_trace(go.Scatter(
    x=x_days[mask_obs], y=observed_net[mask_obs],
    mode='markers',
    marker=dict(color='red', size=12, symbol='circle'),
    name='Actual Data'
))

fig.update_layout(
    title="Cumulative Net ROAS Trajectory",
    xaxis_title="Days Since Install",
    yaxis_title="ROAS (x)",
    template="plotly_white",
    height=500,
    hovermode="x unified",
    xaxis=dict(tickmode='array', tickvals=[7, 14, 28, 90, 180, 360, 720])
)

st.plotly_chart(fig, use_container_width=True)
