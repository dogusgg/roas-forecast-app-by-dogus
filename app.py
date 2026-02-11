import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Net ROAS Prediction Engine", layout="centered")

st.title("🎯 Net ROAS Prediction Engine")
st.caption("Net-Targeted Multiplier · Store Fee Calibrated · Expert Edition")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

# --- 1. Settings ---
st.subheader("1. Profitability & Retention")
c1, c2 = st.columns([1, 2])
with c1:
    GROSS_TO_NET = st.number_input("Net Revenue Factor (0.70 for 30% fee)", 0.0, 1.0, 0.70)
with c2:
    sel_ret_days = st.multiselect("Retention Days", [1, 7, 28], default=[1, 7, 28])

ret_data = {}
cols = st.columns(len(sel_ret_days))
for i, d in enumerate(sorted(sel_ret_days)):
    with cols[i]:
        def_val = {1:0.40, 7:0.20, 28:0.10}.get(d, 0.0)
        ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_val, 0.01)

# --- 2. ROAS Inputs ---
st.subheader("2. ROAS Data Points")
sel_roas_days = st.multiselect("ROAS Days", [1, 3, 7, 14, 28], default=[1, 3, 7, 14, 28])
roas_iap = {}
for d in sorted(sel_roas_days):
    def_iap = {1:0.02, 3:0.05, 7:0.10, 14:0.16, 28:0.25}.get(d, 0.0)
    roas_iap[d] = st.number_input(f"Day {d} IAP ROAS (Gross)", 0.0, 10.0, def_iap, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])

run = st.button("🚀 RUN NET FORECAST", use_container_width=True, type="primary")

if not run: st.stop()

# ==========================================
# 🔥 NET-CALIBRATED MATHEMATICAL MODEL
# ==========================================

def calculate_retention_score(ret_dict):
    # D28 ağırlıklı kalite skoru
    return (0.6 * ret_dict.get(28, 0.1)) + (0.3 * ret_dict.get(7, 0.2)) + (0.1 * ret_dict.get(1, 0.4))

def net_targeted_forecast(days, roas, ret_score, net_factor):
    mask = roas > 0
    last_d, last_r = days[mask][-1], roas[mask][-1]
    
    # --- NET MULTIPLIER LOGIC ---
    # Hedef Net ROAS'ları vurmak için Brüt çarpanları (Gross Multipliers)
    # Vaka 1 (D7) için ~12.3x Brüt -> 0.10 * 12.3 * 0.70 * 0.86 (hill) = ~0.74 Net
    # Vaka 2 (D28) için ~5.8x Brüt -> 0.25 * 5.8 * 0.70 * 0.86 (hill) = ~0.87 Net
    
    base_time_mult = 36.5 * (last_d ** -0.55) 
    
    # Retention Elasticity (Vaka 3 için %25+ Net artışı sağlar)
    ret_factor = (ret_score / 0.16) ** 1.1 
    
    final_mult = base_time_mult * ret_factor
    ceiling = last_r * final_mult
    
    # Hill Dynamics
    k, h = 85.0, 1.2
    forecast_gross = ceiling * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)
    
    return forecast_gross * net_factor # Doğrudan Net döndürüyoruz

# Execution
ret_score = calculate_retention_score(ret_data)
net_forecast = net_targeted_forecast(x_days, y_iap, ret_score, GROSS_TO_NET)

# ==========================================
# 3. RESULTS
# ==========================================

st.divider()
st.metric("D360 NET ROAS FORECAST", f"{net_forecast[3]:.2f}x")

df_res = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "NET ROAS Forecast": net_forecast.round(3)
})
st.dataframe(df_res, use_container_width=True, hide_index=True)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_forecast, mode='lines+markers', name='Net Forecast', line=dict(color='#FF4B4B', width=4)))
fig.update_layout(title="Predicted Net ROAS Curve", template="plotly_white", xaxis_title="Days", yaxis_title="Net ROAS")
st.plotly_chart(fig, use_container_width=True)
