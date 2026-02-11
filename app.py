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
st.caption("Power-Law Time Decay · Retention Elasticity Model · Conservative Calibration")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

# ==========================================
# 1. INPUT SECTION
# ==========================================

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

st.subheader("2. Retention Metrics")
st.info("💡 Model, D28 Retention verisine yüksek ağırlık verir.")
ret_days_options = [1, 3, 7, 14, 28, 60]
sel_ret_days = st.multiselect("Select Available Retention Days", ret_days_options, default=[1, 7, 28])

ret_data = {}
cols = st.columns(len(sel_ret_days)) if len(sel_ret_days) > 0 else [st.empty()]

for i, d in enumerate(sorted(sel_ret_days)):
    with cols[i]:
        def_val = {1: 0.40, 7: 0.20, 28: 0.10}.get(d, 0.0)
        ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_val, 0.01)

st.subheader("3. ROAS Data Points")
roas_days_options = [1, 3, 7, 14, 28, 45, 60, 90]
sel_roas_days = st.multiselect("Select Available ROAS Days", roas_days_options, default=[1, 3, 7, 14, 28])

roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1: 0.02, 3: 0.05, 7: 0.10, 14: 0.16, 28: 0.25}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 10.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

generate = st.button("🚀 RUN FORECAST MODEL", use_container_width=True)

if not generate: st.stop()

# ==========================================
# 2. CALIBRATED MATHEMATICAL MODEL
# ==========================================

def calculate_retention_score(ret_dict):
    d1, d7, d28 = ret_dict.get(1, 0.4), ret_dict.get(7, 0.2), ret_dict.get(28, 0.1)
    return (0.6 * d28) + (0.3 * d7) + (0.1 * d1)

def projected_hill_function(days_array, roas_array, ret_score, mode="iap"):
    mask = roas_array > 0
    if np.sum(mask) == 0:
        return np.zeros(len(FUTURE_DAYS)), np.zeros(len(FUTURE_DAYS)), np.zeros(len(FUTURE_DAYS))
    
    last_day = days_array[mask][-1]
    last_roas = roas_array[mask][-1]
    
    # --- MULTIPLIER LOGIC ---
    # IAP Base: y = 36.5 * x^-0.55
    # AD Base: IAP'nin yarısı çarpan (felaket senaryosu)
    base_mult = 36.5 * (last_day ** -0.55)
    if mode == "ad":
        base_mult = base_mult * 0.5
        
    ret_factor = (ret_score / 0.16) ** 1.1
    final_mult = base_mult * ret_factor
    
    # --- IAP DOWN-SCALE ---
    # IAP forecastlerini %10 aşağı çekme revizesi
    if mode == "iap":
        final_mult = final_mult * 0.90

    ceiling_roas = last_roas * final_mult
    
    k, h = 85.0, 1.2
    forecast_values = ceiling_roas * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)
    
    uncertainty = 0.20 * (7 / last_day) ** 0.5
    return forecast_values, forecast_values * (1 - uncertainty), forecast_values * (1 + uncertainty)

# ==========================================
# 3. EXECUTION & RESULTS
# ==========================================

ret_score = calculate_retention_score(ret_data)
iap_pred, iap_low, iap_high = projected_hill_function(x_days, y_iap, ret_score, mode="iap")
ad_pred, ad_low, ad_high = projected_hill_function(x_days, y_ad, ret_score, mode="ad")

net_pred = (iap_pred * GROSS_TO_NET) + ad_pred
net_low = (iap_low * GROSS_TO_NET) + ad_low
net_high = (iap_high * GROSS_TO_NET) + ad_high

st.divider()

c1, c2, c3 = st.columns(3)
with c1: st.metric("D360 Forecast (Net)", f"{net_pred[3]:.2f}x")
with c2: st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x")
with c3: st.metric("Implied LTV Multiplier", f"{(net_pred[3] / ((y_iap[-1]*GROSS_TO_NET)+y_ad[-1]) if (y_iap[-1]+y_ad[-1])>0 else 0):.1f}x")

df_res = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "IAP Forecast": iap_pred.round(3),
    "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3)
})
st.dataframe(df_res, use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_pred, mode='lines+markers', line=dict(color='#0068C9', width=4), name='Net Forecast'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=iap_pred * GROSS_TO_NET, mode='lines', line=dict(color='#29B09D', dash='dash'), name='Net IAP'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=ad_pred, mode='lines', line=dict(color='#FFBD45', dash='dot'), name='Ad ROAS'))
st.plotly_chart(fig, use_container_width=True)
