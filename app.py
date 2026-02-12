import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Predictor", layout="centered")

# --- UI & CSS ---
st.title("🎯 ROAS Predictor")
st.markdown("""
<style>
div.stButton > button:first-child:not(:disabled) {
    background-color: #FF4B4B; color: white; font-size: 20px; font-weight: bold;
    border-radius: 10px; padding: 15px 0; border: none;
}
</style>
""", unsafe_allow_html=True)

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

# ==========================================
# 1. INPUT SECTION
# ==========================================

st.subheader("1. Profitability & Model Settings")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    fee_mode = st.selectbox("Platform Fees", ["Standard (30%)", "SMB (15%)", "Custom"])
    GROSS_TO_NET = 0.70 if fee_mode == "Standard (30%)" else 0.85 if fee_mode == "SMB (15%)" else 0.70
with c2:
    # Tahminlerin düşük gelmesini engelleyen ana kontrol
    growth_agg = st.select_slider("Growth Aggressiveness", options=["Low", "Standard", "High", "Aggressive"], value="Standard")
    # Katsayıyı belirle (P değerini shift eder)
    agg_map = {"Low": 0.20, "Standard": 0.32, "High": 0.42, "Aggressive": 0.52}
    base_p = agg_map[growth_agg]
with c3:
    st.write("") # Spacer
    st.info(f"Base Growth Exponent: **{base_p}**")

st.subheader("2. Retention Metrics")
ret_days_options = [1, 3, 7, 14, 28, 45, 60]
sel_ret_days = st.multiselect("Select Available Retention Days", ret_days_options, default=[1, 7, 28, 60])

ret_data = {}
if sel_ret_days:
    cols = st.columns(len(sel_ret_days))
    for i, d in enumerate(sorted(sel_ret_days)):
        with cols[i]:
            def_val = {1: 0.40, 7: 0.20, 28: 0.10, 60: 0.07}.get(d, 0.0)
            ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_val, 0.01)

st.subheader("3. ROAS Data Points")
roas_days_options = [1, 3, 7, 14, 28, 45, 60]
sel_roas_days = st.multiselect("Select Available ROAS Days", roas_days_options, default=[1, 7, 28, 60])

roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1: 0.02, 7: 0.10, 28:0.25, 60: 0.40}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 10.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

is_disabled = (np.sum(y_iap > 0) < 3) and (np.sum(y_ad > 0) < 3)
generate = st.button("🚀 RUN FORECAST MODEL", disabled=is_disabled, use_container_width=True)

if is_disabled or not generate:
    if is_disabled: st.warning("⚠️ Aktivasyon için en az 3 veri noktası gereklidir.")
    st.stop()

# ==========================================
# 2. CORE MATHEMATICS (BOOSTED)
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

def anchored_power_law(days_array, roas_array, perf_score, base_p, mode="iap"):
    mask = roas_array > 0
    if np.sum(mask) == 0: return np.zeros(len(FUTURE_DAYS))
    
    last_day = days_array[mask][-1]
    last_roas = roas_array[mask][-1]
    
    # PERFORMANS AYARI: Baz üs değeri üzerine performans ekle
    # base_p (0.32) + perf_score katkısı (0.15 * (perf - 1))
    p = base_p + (perf_score - 1.0) * 0.5
    
    # AD tahmini genellikle IAP'den daha yavaş büyür
    if mode == "ad": p *= 0.70 
    
    p = max(0.05, min(0.65, p)) # Güvenlik sınırı
    
    return last_roas * (FUTURE_DAYS / last_day) ** p

# ==========================================
# 3. EXECUTION & RESULTS
# ==========================================

perf_score = calculate_performance_score(ret_data)
iap_pred = anchored_power_law(x_days, y_iap, perf_score, base_p, mode="iap")
ad_pred = anchored_power_law(x_days, y_ad, perf_score, base_p, mode="ad")
net_pred = (iap_pred * GROSS_TO_NET) + ad_pred

# Hata Payı Bandı
uncertainty = 0.12 * (7 / x_days[y_iap+y_ad>0][-1]) ** 0.5
net_low, net_high = net_pred * (1 - uncertainty), net_pred * (1 + uncertainty)

st.divider()
c1, c2, c3 = st.columns(3)
with c1: st.metric("D360 Forecast (Net)", f"{net_pred[3]:.2f}x", delta=f"Perf Score: {perf_score:.2f}")
with c2: st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x")
with c3:
    last_actual = (y_iap[y_iap>0][-1] * GROSS_TO_NET) + (y_ad[y_ad>0][-1] if np.any(y_ad>0) else 0)
    multiplier = (net_pred[3] / last_actual) if last_actual > 0 else 0
    st.metric("LTV Growth Multiplier", f"{multiplier:.1f}x (Last to D360)")

# Tablo ve Grafik
df_res = pd.DataFrame({
    "Day": FUTURE_DAYS, "IAP Forecast": iap_pred.round(3), "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3), "Cons. (-15%)": net_low.round(3), "Opt. (+15%)": net_high.round(3)
})
st.dataframe(df_res, use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]), y=np.concatenate([net_high, net_low[::-1]]), fill='toself', fillcolor='rgba(0, 104, 201, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_pred, mode='lines+markers', line=dict(color='#0068C9', width=4), name='Net Forecast'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=iap_pred * GROSS_TO_NET, mode='lines', line=dict(color='#29B09D', dash='dash'), name='Net IAP ROAS'))

if np.any(y_iap > 0):
    fig.add_trace(go.Scatter(x=x_days[y_iap>0], y=(y_iap*GROSS_TO_NET + y_ad)[y_iap>0], mode='markers', marker=dict(color='red', size=10), name='Actual Data'))

fig.update_layout(title="ROAS Trajectory (Aggressive Tuning Enabled)", template="plotly_white", height=500)
st.plotly_chart(fig, use_container_width=True)
