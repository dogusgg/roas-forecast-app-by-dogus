import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Predictor", layout="centered")

# --- HEADER & FIXED CSS ---
st.title("🎯 ROAS Predictor")
st.markdown("""
<style>
/* Sadece aktif (disabled olmayan) butonu kırmızı yap */
div.stButton > button:first-child:not(:disabled) {
    background-color: #FF4B4B;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 15px 0;
    border: none;
}

/* Inaktif (disabled) butonu beyaz ve gri çerçeveli yap */
div.stButton > button:disabled {
    background-color: white !important;
    color: #bcbcbc !important;
    border: 1px solid #bcbcbc !important;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 15px 0;
    cursor: not-allowed;
}
</style>
""", unsafe_allow_html=True)
st.caption("Power-Law Time Decay · Precise Calibration · Fixed UI State")

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
ret_days_options = [1, 3, 7, 14, 28, 45, 60]
sel_ret_days = st.multiselect("Select Available Retention Days", ret_days_options, default=[1, 7, 28])

ret_data = {}
cols = st.columns(len(sel_ret_days)) if len(sel_ret_days) > 0 else [st.empty()]
for i, d in enumerate(sorted(sel_ret_days)):
    with cols[i]:
        def_val = {1: 0.40, 7: 0.20, 28: 0.10}.get(d, 0.0)
        ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_val, 0.01)

st.subheader("3. ROAS Data Points")
roas_days_options = [1, 3, 7, 14, 28, 45, 60]
sel_roas_days = st.multiselect("Select Available ROAS Days", roas_days_options, default=[1, 3, 7, 14, 28])

roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1: 0.00, 3: 0.00, 7: 0.00, 14: 0.00, 28: 0.00}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 10.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

# --- BUTON AKTİVASYON MANTIĞI ---
iap_point_count = np.sum(y_iap > 0)
ad_point_count = np.sum(y_ad > 0)
is_disabled = (iap_point_count < 3) and (ad_point_count < 3)

# Buton oluşturma (CSS'e göre stil değişecek)
generate = st.button("🚀 RUN FORECAST MODEL", disabled=is_disabled, use_container_width=True)

if is_disabled:
    st.warning("⚠️ Aktivasyon için IAP veya AD serisinden en az birinde 3 adet veri girişi gereklidir.")
    st.stop()

if not generate:
    st.stop()

# ==========================================
# 2. CALIBRATED MATHEMATICAL MODEL
# ==========================================

def calculate_retention_score(ret_dict):
    base_weights = {1: 0.05, 3: 0.05, 7: 0.10, 14: 0.20, 28: 0.25, 45: 0.15, 60: 0.20}
    score, total_w = 0, 0
    for d, val in ret_dict.items():
        w = base_weights.get(d, 0.1)
        score += val * w
        total_w += w
    return score / total_w if total_w > 0 else 0.16

def projected_hill_function(days_array, roas_array, ret_score, mode="iap"):
    mask = roas_array > 0
    if np.sum(mask) == 0: return np.zeros(len(FUTURE_DAYS))
    
    last_day, last_roas = days_array[mask][-1], roas_array[mask][-1]
    
    # 720/360 Oranı Kalibrasyonu (0.16 -> ~1.1 | 0.205 -> ~1.2)
    h = 0.45 + (ret_score - 0.16) * 8.8
    k = 450.0 + (ret_score - 0.16) * 1500
    
    base_mult = 34.5 * (last_day ** -0.55)
    
    if mode == "ad":
        base_mult = base_mult * 0.82 # AD boost/penalty ayarı
    
    ret_factor = (ret_score / 0.16) ** 1.3
    final_mult = base_mult * ret_factor
    
    if mode == "iap":
        final_mult = final_mult * 0.90 # IAP Penalty

    ceiling_roas = last_roas * final_mult
    def hill(t, k_val, h_val): return (t**h_val) / (k_val**h_val + t**h_val)
    
    scale = last_roas / hill(last_day, k, h)
    return scale * hill(FUTURE_DAYS, k, h)

# ==========================================
# 3. EXECUTION & VISUALS
# ==========================================

ret_score = calculate_retention_score(ret_data)
iap_pred = projected_hill_function(x_days, y_iap, ret_score, mode="iap")
ad_pred = projected_hill_function(x_days, y_ad, ret_score, mode="ad")
net_pred = (iap_pred * GROSS_TO_NET) + ad_pred

uncertainty = 0.18 * (7 / x_days[y_iap+y_ad>0][-1]) ** 0.5
net_low, net_high = net_pred * (1 - uncertainty), net_pred * (1 + uncertainty)

st.divider()
c1, c2, c3 = st.columns(3)
with c1: st.metric("D360 Forecast (Net)", f"{net_pred[3]:.2f}x", delta=f"Range: {net_low[3]:.2f}-{net_high[3]:.2f}")
with c2: st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x")
with c3:
    actual_net = (y_iap[-1] * GROSS_TO_NET) + y_ad[-1]
    st.metric("Implied LTV Multiplier", f"{(net_pred[3] / actual_net if actual_net > 0 else 0):.1f}x")

st.dataframe(pd.DataFrame({
    "Day": FUTURE_DAYS, "IAP Forecast": iap_pred.round(3), "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3), "Conservative": net_low.round(3), "Optimistic": net_high.round(3)
}), use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]), y=np.concatenate([net_high, net_low[::-1]]), fill='toself', fillcolor='rgba(0, 104, 201, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval', hoverinfo="skip"))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_pred, mode='lines+markers', line=dict(color='#0068C9', width=4), name='Net Forecast'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=iap_pred * GROSS_TO_NET, mode='lines', line=dict(color='#29B09D', dash='dash'), name='Net IAP'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=ad_pred, mode='lines', line=dict(color='#FFBD45', dash='dot'), name='Ad ROAS'))

if np.any(y_iap > 0):
    fig.add_trace(go.Scatter(x=x_days[y_iap>0], y=y_iap[y_iap>0], mode='markers', marker=dict(color='#29B09D', size=10, symbol='x'), name='IAP_Observed'))
if np.any(y_ad > 0):
    fig.add_trace(go.Scatter(x=x_days[y_ad>0], y=y_ad[y_ad>0], mode='markers', marker=dict(color='#FFBD45', size=10, symbol='circle'), name='AD_Observed'))

fig.update_layout(title="Cumulative Net ROAS Trajectory", template="plotly_white", height=500, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
