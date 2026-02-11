import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Prediction Engine", layout="wide")

st.title("📈 ROAS Prediction Engine")
st.caption("Final Edition · Net-Targeted · IAP & AD Integrated")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

# ==========================================
# 1. INPUT SECTION (TAM UI)
# ==========================================
col_top1, col_top2 = st.columns([1, 2])

with col_top1:
    st.subheader("1. Profitability")
    fee_mode = st.selectbox("Store Fee Mode", ["Standard (30%)", "SMB (15%)", "Custom"])
    if fee_mode == "Standard (30%)": GROSS_TO_NET = 0.70
    elif fee_mode == "SMB (15%)": GROSS_TO_NET = 0.85
    else: GROSS_TO_NET = st.number_input("Custom Factor", 0.0, 1.0, 0.70)

with col_top2:
    st.subheader("2. Retention Metrics")
    sel_ret_days = st.multiselect("Select Retention Days", [1, 7, 28], default=[1, 7, 28])
    ret_data = {}
    r_cols = st.columns(len(sel_ret_days))
    for i, d in enumerate(sorted(sel_ret_days)):
        with r_cols[i]:
            def_v = {1:0.40, 7:0.20, 28:0.10}.get(d, 0.0)
            ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_v, 0.01)

st.subheader("3. ROAS Data Points")
sel_roas_days = st.multiselect("Select ROAS Days", [1, 3, 7, 14, 28], default=[1, 3, 7, 14, 28])

roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1:0.02, 3:0.05, 7:0.10, 14:0.16, 28:0.25}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS (Gross)", 0.0, 10.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

run = st.button("🚀 GENERATE FORECAST", use_container_width=True, type="primary")

if not run: st.stop()

# ==========================================
# 2. CALIBRATED ENGINE
# ==========================================

def calculate_retention_score(ret_dict):
    return (0.6 * ret_dict.get(28, 0.1)) + (0.3 * ret_dict.get(7, 0.2)) + (0.1 * ret_dict.get(1, 0.4))

def hill_forecast(days, roas, ret_score, net_factor, is_iap=True):
    mask = roas > 0
    if np.sum(mask) == 0: return np.zeros(len(FUTURE_DAYS))
    last_d, last_r = days[mask][-1], roas[mask][-1]
    
    # Net Hedeflere (0.75, 0.85, 1.05) Göre Brüt Çarpan Kalibrasyonu
    # IAP için store fee telafisi eklenir, AD için eklenmez.
    base_mult = 36.5 * (last_d ** -0.55)
    ret_factor = (ret_score / 0.16) ** 1.1
    
    final_mult = base_time_mult = base_mult * ret_factor
    ceiling = last_r * final_mult
    
    k, h = 85.0, 1.2 # Stabil Hill Dinamikleri
    forecast = ceiling * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)
    
    # IAP ise net_factor (0.70) uygula, AD ise dokunma.
    return forecast * net_factor if is_iap else forecast

# Hesaplamalar
ret_score = calculate_retention_score(ret_data)
iap_net_pred = hill_forecast(x_days, y_iap, ret_score, GROSS_TO_NET, is_iap=True)
ad_net_pred = hill_forecast(x_days, y_ad, ret_score, 1.0, is_iap=False) # Ad ROAS factor = 1.0

total_net_pred = iap_net_pred + ad_net_pred
total_net_low = total_net_pred * 0.85 # %15 Güven Aralığı
total_net_high = total_net_pred * 1.15

# ==========================================
# 3. RESULTS DISPLAY
# ==========================================
st.divider()
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1: st.metric("D360 Net ROAS", f"{total_net_pred[3]:.2f}x")
with kpi2: st.metric("D180 Net ROAS", f"{total_net_pred[2]:.2f}x")
with kpi3: st.metric("Net Quality Score", f"{ret_score:.3f}")

# Table
df_res = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "IAP Net Forecast": iap_net_pred.round(3),
    "AD ROAS Forecast": ad_net_pred.round(3),
    "TOTAL NET ROAS": total_net_pred.round(3)
})
st.dataframe(df_res, use_container_width=True, hide_index=True)

# Graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=total_net_pred, mode='lines+markers', name='Total Net Forecast', line=dict(color='#FF4B4B', width=4)))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=total_net_high, line=dict(width=0), showlegend=False, name='Upper'))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=total_net_low, line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 75, 75, 0.1)', showlegend=False, name='Lower'))
fig.update_layout(title="Net ROAS Trajectory (Store Fees Deducted from IAP)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
