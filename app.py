import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Predictor", layout="centered")

# --- UI & CSS ---
st.markdown("""
<style>
div.stButton > button:first-child:not(:disabled) {
    background-color: #FF4B4B; color: white; font-size: 20px; font-weight: bold;
    border-radius: 10px; padding: 15px 0; border: none;
}
div.stButton > button:disabled {
    background-color: white !important; color: #bcbcbc !important;
    border: 1px solid #bcbcbc !important; font-size: 20px; font-weight: bold;
    border-radius: 10px; padding: 15px 0;
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
    GROSS_TO_NET = 0.70 if fee_mode == "Standard (30%)" else 0.85 if fee_mode == "SMB (15%)" else st.number_input("Custom Net Factor", 0.0, 1.0, 0.70)

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
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 10.0, {1: 0.02, 3: 0.05, 7: 0.10, 14:0.16, 28:0.25}.get(d, 0.0), 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

is_disabled = (np.sum(y_iap > 0) < 3) and (np.sum(y_ad > 0) < 3)
generate = st.button("🚀 RUN FORECAST MODEL", disabled=is_disabled, use_container_width=True)

if is_disabled:
    st.warning("⚠️ Aktivasyon için IAP veya AD serisinden en az birinde 3 adet veri girişi gereklidir.")
    st.stop()
if not generate: st.stop()

# ==========================================
# 2. 🔥 RELATIVE PERFORMANCE MODEL (D60 FIX)
# ==========================================

def calculate_performance_score(ret_dict):
    if not ret_dict: return 1.0 # Baseline
    
    # 1. Baz Değerler (Bu değerler 1.0 performans puanı demektir)
    baselines = {1: 0.40, 3: 0.28, 7: 0.20, 14: 0.14, 28: 0.10, 45: 0.08, 60: 0.07}
    
    # 2. Önem Ağırlıkları (Geç günler daha etkili)
    importance = {1: 1, 3: 2, 7: 5, 14: 8, 28: 15, 45: 18, 60: 25}
    
    weighted_perf = 0
    total_importance = 0
    
    for day, val in ret_dict.items():
        base = baselines.get(day, 0.10)
        imp = importance.get(day, 10)
        
        # Günlük performans (Örn: D60 %9 girildiyse -> 0.09 / 0.07 = 1.28)
        perf_ratio = val / base if base > 0 else 1.0
        
        weighted_perf += perf_ratio * imp
        total_importance += imp
    
    # Normalize edilmiş genel performans skoru (1.0 = baseline, >1.0 = daha iyi)
    return weighted_perf / total_importance

def power_law_projection(days_array, roas_array, perf_score, mode="iap"):
    mask = roas_array > 0
    if np.sum(mask) == 0: return np.zeros(len(FUTURE_DAYS))
    
    last_day, last_roas = days_array[mask][-1], roas_array[mask][-1]
    
    # Baz büyüme üssü (p)
    # p = 0.14 baseline (~1.10 ratio)
    # perf_score 1.28 olursa p artar ve ratio yükselir
    p = 0.14 + (perf_score - 1.0) * 0.3
    p = max(0.05, min(0.40, p)) 
    
    decay = (FUTURE_DAYS / last_day) ** p
    if mode == "ad": decay *= 0.6
    
    return last_roas * decay

# ==========================================
# 3. EXECUTION
# ==========================================

perf_score = calculate_performance_score(ret_data)
iap_pred = power_law_projection(x_days, y_iap, perf_score, mode="iap")*3
ad_pred = power_law_projection(x_days, y_ad, perf_score, mode="ad")
net_pred = (iap_pred * GROSS_TO_NET) + ad_pred

# Belirsizlik
uncertainty = 0.12 * (7 / x_days[y_iap+y_ad>0][-1]) ** 0.5
net_low, net_high = net_pred * (1 - uncertainty), net_pred * (1 + uncertainty)

# ==========================================
# 4. RESULTS
# ==========================================

st.divider()
c1, c2, c3 = st.columns(3)
with c1: st.metric("D360 Forecast (Net)", f"{net_pred[3]:.2f}x", delta=f"Score: {perf_score:.2f}")
with c2: st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x")
with c3:
    last_actual = (y_iap[y_iap>0][-1] * GROSS_TO_NET) if np.any(y_iap>0) else 0
    last_actual += y_ad[y_ad>0][-1] if np.any(y_ad>0) else 0
    st.metric("Implied LTV Multiplier", f"{(net_pred[3] / last_actual if last_actual > 0 else 0):.1f}x")

# 720/360 oranını göstermek için küçük bir info
ratio_720_360 = iap_pred[4] / iap_pred[3] if iap_pred[3] > 0 else 0
st.info(f"📈 720/360 Growth Ratio: **{ratio_720_360:.3f}** (Affected by Retention Score)")

st.dataframe(pd.DataFrame({
    "Day": FUTURE_DAYS, "IAP Forecast": iap_pred.round(3), "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3), "Conservative": net_low.round(3), "Optimistic": net_high.round(3)
}), use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_pred, mode='lines+markers', line=dict(color='#0068C9', width=4), name='Net Forecast'))
fig.add_trace(go.Scatter(x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]), y=np.concatenate([net_high, net_low[::-1]]), fill='toself', fillcolor='rgba(0, 104, 201, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
fig.update_layout(title="ROAS Trajectory (Relative Performance Model)", template="plotly_white", height=450)
st.plotly_chart(fig, use_container_width=True)






