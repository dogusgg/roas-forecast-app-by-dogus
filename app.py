import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Predictor", layout="centered")

# --- HEADER ---
st.title("🎯 ROAS Predictor")
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
st.caption("Power-Law Time Decay · Retention Elasticity Model · Final Stable Build")

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
        GROSS_TO_NET = st.number_input("Custom Net Factor", 0.0, 1.0, 0.70)

st.subheader("2. Retention Metrics")
sel_ret_days = st.multiselect("Select Available Retention Days", [1, 7, 28], default=[1, 7, 28])
ret_data = {}
cols = st.columns(3)
for i, d in enumerate([1, 7, 28]):
    with cols[i]:
        if d in sel_ret_days:
            def_val = {1: 0.40, 7: 0.20, 28: 0.10}.get(d, 0.0)
            ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_val, 0.01)

st.subheader("3. ROAS Data Points")
sel_roas_days = st.multiselect("Select Available ROAS Days", [1, 3, 7, 14, 28], default=[1, 3, 7, 14, 28])
roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1: 0.00, 3: 0.00, 7: 0.00, 14: 0.00, 28: 0.00}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 1.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 1.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap.get(d, 0.0) for d in x_days])
y_ad = np.array([roas_ad.get(d, 0.0) for d in x_days])

# --- BUTON KONTROLÜ ---
total_valid_points = np.sum(y_iap > 0) + np.sum(y_ad > 0)
btn_disabled = total_valid_points < 3
generate = st.button("🚀 RUN FORECAST MODEL", disabled=btn_disabled, use_container_width=True)

if not generate:
    if btn_disabled:
        st.warning("⚠️ Aktivasyon için en az 3 ROAS girişi yapmalısın.")
    st.stop()

# ==========================================
# 2. CALIBRATED MATHEMATICAL MODEL
# ==========================================

def calculate_retention_score(ret_dict):
    d1, d7, d28 = ret_dict.get(1, 0.4), ret_dict.get(7, 0.2), ret_dict.get(28, 0.1)
    return (0.6 * d28) + (0.3 * d7) + (0.1 * d1)

def projected_hill_function(days_array, roas_array, ret_score, mode="iap"):
    mask = roas_array > 0
    if np.sum(mask) == 0: return np.zeros(len(FUTURE_DAYS))
    last_day, last_roas = days_array[mask][-1], roas_array[mask][-1]
    
    base_mult = 36.5 * (last_day ** -0.55)
    ret_factor = (ret_score / 0.16) ** 1.3
    
    if mode == "ad":
        # Reklam çarpanı iyileştirildi
        final_mult = base_mult * ret_factor * (0.75 + (ret_score - 0.16) * 2)
    else:
        # IAP Penalty %10
        final_mult = base_mult * ret_factor * 0.90 

    ceiling_roas = last_roas * final_mult
    # D720/D360 oranı ayarı (1.1x - 1.2x)
    h = 1.2 + (ret_score - 0.16) * 5.5
    k = 85.0 + (ret_score - 0.16) * 150
    return ceiling_roas * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)

# ==========================================
# 3. EXECUTION
# ==========================================

ret_score = calculate_retention_score(ret_data)
iap_pred = projected_hill_function(x_days, y_iap, ret_score, mode="iap")
ad_pred = projected_hill_function(x_days, y_ad, ret_score, mode="ad")
net_pred = (iap_pred * GROSS_TO_NET) + ad_pred

uncertainty = 0.15 * (7 / x_days[y_iap+y_ad>0][-1]) ** 0.5
net_low, net_high = net_pred * (1 - uncertainty), net_pred * (1 + uncertainty)

st.divider()
c1, c2, c3 = st.columns(3)
with c1: st.metric("D360 Forecast (Net)", f"{net_pred[3]:.2f}x", delta=f"Range: {net_low[3]:.2f}-{net_high[3]:.2f}")
with c2: st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x")

last_obs_net = (y_iap[-1] * GROSS_TO_NET) + y_ad[-1]
with c3: st.metric("Implied LTV Multiplier", f"{(net_pred[3]/last_obs_net if last_obs_net > 0 else 0):.1f}x")

st.dataframe(pd.DataFrame({
    "Day": FUTURE_DAYS, "IAP Forecast": iap_pred.round(3), "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3), "Conservative": net_low.round(3), "Optimistic": net_high.round(3)
}), use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]), y=np.concatenate([net_high, net_low[::-1]]), fill='toself', fillcolor='rgba(0, 104, 201, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval', hoverinfo="skip"))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_pred, mode='lines+markers', line=dict(color='#0068C9', width=4), name='Net Forecast'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=iap_pred * GROSS_TO_NET, mode='lines', line=dict(color='#29B09D', dash='dash'), name='Net IAP Contribution'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=ad_pred, mode='lines', line=dict(color='#FFBD45', dash='dot'), name='Ad Forecast'))

if np.any(y_iap > 0): fig.add_trace(go.Scatter(x=x_days[y_iap>0], y=y_iap[y_iap>0]*GROSS_TO_NET, mode='markers', marker=dict(color='#29B09D', size=8, symbol='x'), name='IAP Observed (Net)'))
if np.any(y_ad > 0): fig.add_trace(go.Scatter(x=x_days[y_ad>0], y=y_ad[y_ad>0], mode='markers', marker=dict(color='#FFBD45', size=8, symbol='x'), name='Ad Observed'))

fig.update_layout(title="Cumulative Net ROAS Trajectory", template="plotly_white", height=500, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
