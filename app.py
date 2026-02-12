import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Predictor", layout="centered")

# --- UI & CSS (İlk Kodundaki Stil) ---
st.title("🎯 ROAS Predictor")
st.markdown("""
<style>
div.stButton > button:first-child:not(:disabled) {
    background-color: #FF4B4B;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 15px 0;
    border: none;
}
div.stButton > button:disabled {
    background-color: white !important;
    color: #bcbcbc !important;
    border: 1px solid #bcbcbc !important;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 15px 0;
}
</style>
""", unsafe_allow_html=True)

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
        GROSS_TO_NET = st.number_input("Custom Net", 0.0, 1.0, 0.70)

st.subheader("2. Retention Metrics")
ret_days_options = [1, 3, 7, 14, 28, 45, 60]
sel_ret_days = st.multiselect("Select Available Retention Days", ret_days_options, default=[1, 7, 28])

ret_data = {}
if sel_ret_days:
    cols = st.columns(len(sel_ret_days))
    for i, d in enumerate(sorted(sel_ret_days)):
        with cols[i]:
            def_val = {1: 0.40, 7: 0.20, 28: 0.10, 60: 0.07}.get(d, 0.0)
            ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_val, 0.01)

st.subheader("3. ROAS Data Points")
roas_days_options = [1, 3, 7, 14, 28, 45, 60]
sel_roas_days = st.multiselect("Select Available ROAS Days", roas_days_options, default=[1, 3, 7, 14, 28])

roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1: 0.02, 3: 0.05, 7: 0.10, 14: 0.16, 28: 0.25}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 5.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 5.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

# Aktivasyon Mantığı
is_disabled = (np.sum(y_iap > 0) < 3) and (np.sum(y_ad > 0) < 3)
generate = st.button("🚀 RUN FORECAST MODEL", disabled=is_disabled, use_container_width=True)

if is_disabled:
    st.warning("⚠️ Forecast için IAP_ROAS veya AD_ROAS en az birinde 3 adet veri gereklidir.")
    st.stop()
if not generate: st.stop()

# ==========================================
# 2. CORE MATHEMATICS (FIXED & ANCHORED)
# ==========================================

def calculate_performance_score(ret_dict):
    if not ret_dict: return 1.0
    baselines = {1: 0.40, 3: 0.28, 7: 0.20, 14: 0.14, 28: 0.10, 45: 0.08, 60: 0.07}
    importance = {1: 1, 3: 2, 7: 5, 14: 8, 28: 15, 45: 18, 60: 25}
    weighted_perf, total_imp = 0, 0
    for d, val in ret_dict.items():
        base = baselines.get(d, 0.10)
        imp = importance.get(d, 10)
        weighted_perf += (val / base if base > 0 else 1.0) * imp
        total_imp += imp
    return weighted_perf / total_imp

def anchored_power_law(days_array, roas_array, perf_score, mode="iap"):
    mask = roas_array > 0
    if np.sum(mask) == 0: return np.zeros(len(FUTURE_DAYS))
    
    last_day = days_array[mask][-1]
    last_roas = roas_array[mask][-1]
    
    # p katsayısı (büyüme hızı) perf_score ile belirlenir
    p = 0.22 + (perf_score - 1.0) * 0.66
    p = max(0.08, min(0.99, p))
    
    if mode == "ad": p *= 0.88 # AD doygunluğu daha hızlıdır
    
    # Sıçrama olmaması için projeksiyon son girilen veriden (anchor) başlar
    return last_roas * (FUTURE_DAYS / last_day) ** p

# ==========================================
# 3. EXECUTION
# ==========================================

perf_score = calculate_performance_score(ret_data)
iap_pred = anchored_power_law(x_days, y_iap, perf_score, mode="iap")
ad_pred = anchored_power_law(x_days, y_ad, perf_score, mode="ad")

net_pred = (iap_pred * GROSS_TO_NET) + ad_pred

# Dinamik Belirsizlik Bandı
uncertainty = 0.15 * (7 / x_days[y_iap+y_ad>0][-1]) ** 0.5
net_low, net_high = net_pred * (1 - uncertainty), net_pred * (1 + uncertainty)

# ==========================================
# 4. RESULTS & VISUALIZATION (İlk Kodundaki Düzen)
# ==========================================

st.divider()

col_res1, col_res2, col_res3 = st.columns(3)
with col_res1:
    st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x", 
              delta=f"Range: {net_low[2]:.2f}-{net_high[2]:.2f}")
with col_res2:
    st.metric("D360 Forecast (Net)", f"{net_pred[3]:.2f}x", 
              delta=f"Range: {net_low[3]:.2f}-{net_high[3]:.2f}")
with col_res3:
    st.metric("D720 Forecast (Net)", f"{net_pred[4]:.2f}x", 
              delta=f"Range: {net_low[4]:.2f}-{net_high[4]:.2f}")

# DATA TABLE
df_res = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "IAP Forecast": iap_pred.round(3),
    "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3),
    "Conservative": net_low.round(3),
    "Optimistic": net_high.round(3)
})
st.dataframe(df_res, use_container_width=True, hide_index=True)

# CHART
fig = go.Figure()

# Confidence Tunnel (İlk kodundaki stil)
fig.add_trace(go.Scatter(
    x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]),
    y=np.concatenate([net_high, net_low[::-1]]),
    fill='toself', fillcolor='rgba(0, 104, 201, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval', hoverinfo="skip"
))

# Projeksiyon Çizgileri
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_pred, mode='lines+markers', line=dict(color='#0068C9', width=4), name='Net Blended ROAS Forecast'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=iap_pred * GROSS_TO_NET, mode='lines', line=dict(color='#29B09D', dash='dash'), name='Net IAP ROAS'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=ad_pred, mode='lines', line=dict(color='#FFBD45', dash='dot'), name='Ad ROAS'))

# Gözlemlenen Veriler (Actual Points)
if np.any(y_iap > 0):
    fig.add_trace(go.Scatter(x=x_days[y_iap>0], y=y_iap[y_iap>0], mode='markers', marker=dict(color='red', size=10), name='IAP_Observed'))
if np.any(y_ad > 0):
    fig.add_trace(go.Scatter(x=x_days[y_ad>0], y=y_ad[y_ad>0], mode='markers', marker=dict(color='blue', size=10), name='AD_Observed'))

fig.update_layout(title="Cumulative Net ROAS Trajectory", template="plotly_white", height=550, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# 720/360 Analizi (Alt Bilgi)
ratio_720_360 = iap_pred[4] / iap_pred[3] if iap_pred[3] > 0 else 0
st.caption(f"ℹ️ Retention Score: {perf_score:.2f} | Implied 720/360 Ratio: {ratio_720_360:.3f}")


