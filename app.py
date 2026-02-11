import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Predictor", layout="centered")

# --- UI & CSS ---
st.title("🎯 ROAS Predictor")
st.markdown("""
<style>
/* Sadece aktif butonu kırmızı yap */
div.stButton > button:first-child:not(:disabled) {
    background-color: #FF4B4B;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    padding: 15px 0;
    border: none;
}
/* Inaktif butonu beyaz/gri yap */
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
    GROSS_TO_NET = 0.70 if fee_mode == "Standard (30%)" else 0.85 if fee_mode == "SMB (15%)" else st.number_input("Custom Net Factor", 0.0, 1.0, 0.70)

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
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS", 0.0, 10.0, {1: 0.02, 3: 0.05, 7: 0.10}.get(d, 0.0), 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

# Aktivasyon Mantığı: Herhangi birinde 3 veri varsa çalışır
is_disabled = (np.sum(y_iap > 0) < 3) and (np.sum(y_ad > 0) < 3)
generate = st.button("🚀 RUN FORECAST MODEL", disabled=is_disabled, use_container_width=True)

if is_disabled:
    st.warning("⚠️ Aktivasyon için IAP veya AD serisinden en az birinde 3 adet veri girişi gereklidir.")
    st.stop()
if not generate: st.stop()

# ==========================================
# 2. CALIBRATED MATHEMATICAL MODEL
# ==========================================

def calculate_retention_score(ret_dict):
    # Ağırlıklar: Uzun vade (28-60) toplamın %60'ını oluşturur
    weights = {1: 0.05, 3: 0.05, 7: 0.10, 14: 0.20, 28: 0.25, 45: 0.15, 60: 0.20}
    score, total_w = 0, 0
    for d, val in ret_dict.items():
        w = weights.get(d, 0.1)
        score += val * w
        total_w += w
    return score / total_w if total_w > 0 else 0.16

def power_law_projection(days_array, roas_array, ret_score, mode="iap"):
    mask = roas_array > 0
    if np.sum(mask) == 0: return np.zeros(len(FUTURE_DAYS))
    
    last_day, last_roas = days_array[mask][-1], roas_array[mask][-1]
    
    # 🔥 Büyüme Üssü (p): Bu değer 720/360 oranını belirler (Ratio = 2^p)
    # ret_score 0.16 -> p = 0.14 (Ratio 1.10)
    # ret_score 0.21 -> p = 0.27 (Ratio 1.21)
    p = 0.14 + (ret_score - 0.16) * 2.6
    p = max(0.05, min(0.40, p)) # Güvenlik sınırı
    
    # Uzun vadeli yavaşlama (Time Decay) faktörü
    decay = (FUTURE_DAYS / last_day) ** p
    
    # AD için büyüme hızı biraz daha düşük olabilir (isteğe bağlı ayar)
    if mode == "ad":
        decay = decay * 0.95
        
    return last_roas * decay

# ==========================================
# 3. EXECUTION
# ==========================================

ret_score = calculate_retention_score(ret_data)
iap_pred = power_law_projection(x_days, y_iap, ret_score, mode="iap")
ad_pred = power_law_projection(x_days, y_ad, ret_score, mode="ad")
net_pred = (iap_pred * GROSS_TO_NET) + ad_pred

# Belirsizlik bandı: Veri ne kadar eskiyse o kadar daralır
uncertainty = 0.15 * (7 / x_days[y_iap+y_ad>0][-1]) ** 0.5
net_low, net_high = net_pred * (1 - uncertainty), net_pred * (1 + uncertainty)

# ==========================================
# 4. RESULTS & VISUALS
# ==========================================

st.divider()
c1, c2, c3 = st.columns(3)
with c1: st.metric("D360 Forecast (Net)", f"{net_pred[3]:.2f}x", delta=f"Range: {net_low[3]:.2f}-{net_high[3]:.2f}")
with c2: st.metric("D180 Forecast (Net)", f"{net_pred[2]:.2f}x")
with c3:
    last_actual = (y_iap[y_iap>0][-1] * GROSS_TO_NET) if np.any(y_iap>0) else 0
    last_actual += y_ad[y_ad>0][-1] if np.any(y_ad>0) else 0
    st.metric("Implied LTV Multiplier", f"{(net_pred[3] / last_actual if last_actual > 0 else 0):.1f}x")

st.dataframe(pd.DataFrame({
    "Day": FUTURE_DAYS, "IAP Forecast": iap_pred.round(3), "Ad Forecast": ad_pred.round(3),
    "NET ROAS": net_pred.round(3), "Conservative": net_low.round(3), "Optimistic": net_high.round(3)
}), use_container_width=True, hide_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]), y=np.concatenate([net_high, net_low[::-1]]), fill='toself', fillcolor='rgba(0, 104, 201, 0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_pred, mode='lines+markers', line=dict(color='#0068C9', width=4), name='Net Forecast'))

if np.any(y_iap > 0):
    fig.add_trace(go.Scatter(x=x_days[y_iap>0], y=y_iap[y_iap>0], mode='markers', marker=dict(color='#29B09D', size=10), name='IAP Observed'))
if np.any(y_ad > 0):
    fig.add_trace(go.Scatter(x=x_days[y_ad>0], y=y_ad[y_ad>0], mode='markers', marker=dict(color='#FFBD45', size=10), name='AD Observed'))

fig.update_layout(title="ROAS Trajectory (Power-Law Model)", template="plotly_white", height=450)
st.plotly_chart(fig, use_container_width=True)
