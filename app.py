import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Prediction Engine", layout="wide")

# Görseldeki UI stilini yakalamak için CSS
st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fb; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef; }
    div.stButton > button:first-child { background-color: #ff4b4b; color: white; width: 100%; border-radius: 10px; height: 3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 ROAS Long-Term Forecast")

# ==========================================
# 1. PROFITABILITY & RETENTION (ÜST KISIM)
# ==========================================
st.subheader("1. Profitability Settings")
fee_option = st.selectbox("Platform Fees", ["Standard (30%)", "SMB (15%)", "Custom"], index=0)
if fee_option == "Standard (30%)": GROSS_TO_NET = 0.70
elif fee_option == "SMB (15%)": GROSS_TO_NET = 0.85
else: GROSS_TO_NET = st.number_input("Custom Factor", 0.0, 1.0, 0.70)

st.subheader("2. Retention Metrics")
st.info("💡 Model, D28 Retention verisine yüksek ağırlık verir.")
ret_days = st.multiselect("Select Available Retention Days", [1, 7, 28], default=[1, 7, 28])
ret_data = {}
r_cols = st.columns(3)
for i, d in enumerate([1, 7, 28]):
    if d in ret_days:
        with r_cols[i]:
            def_v = {1:0.40, 7:0.20, 28:0.10}.get(d, 0.0)
            ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_v, 0.01)

# ==========================================
# 2. ROAS DATA POINTS (ORTA KISIM)
# ==========================================
st.subheader("3. ROAS Data Points")
roas_days_list = [1, 3, 7, 14, 28]
sel_roas_days = st.multiselect("Select Available ROAS Days", roas_days_list, default=roas_days_list)

roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1:0.02, 3:0.05, 7:0.10, 14:0.16, 28:0.25}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS (Gross)", 0.0, 10.0, def_iap, 0.01, key=f"iap_{d}")
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01, key=f"ad_{d}")

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap.get(d, 0.0) for d in x_days])
y_ad = np.array([roas_ad.get(d, 0.0) for d in x_days])

run = st.button("🚀 RUN FORECAST MODEL")

if run:
    FUTURE_DAYS = np.array([90, 120, 180, 360, 720])
    
    # --- CALIBRATION ENGINE ---
    def get_forecast(days, roas, ret_dict, net_factor, is_iap=True):
        mask = roas > 0
        if not any(mask): return np.zeros(len(FUTURE_DAYS))
        last_d, last_r = days[mask][-1], roas[mask][-1]
        
        # NET TARGETS: 0.75 (D7), 0.85 (D28), 1.05 (High Ret)
        # Power Law Multiplier: y = 36.5 * x^-0.55
        base_mult = 36.5 * (last_d ** -0.55)
        
        # Retention Quality
        r28, r7, r1 = ret_dict.get(28, 0.1), ret_dict.get(7, 0.2), ret_dict.get(1, 0.4)
        score = (0.6 * r28) + (0.3 * r7) + (0.1 * r1)
        ret_factor = (score / 0.16) ** 1.1
        
        ceiling = last_r * base_mult * ret_factor
        k, h = 85.0, 1.2
        forecast = ceiling * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)
        
        return forecast * net_factor if is_iap else forecast

    iap_net = get_forecast(x_days, y_iap, ret_data, GROSS_TO_NET, True)
    ad_net = get_forecast(x_days, y_ad, ret_data, 1.0, False)
    total_net = iap_net + ad_net
    
    # ==========================================
    # 3. OUTPUT SECTION (GÖRSELDEKİ DÜZEN)
    # ==========================================
    st.divider()
    
    m1, m2, m3 = st.columns(3)
    d360_val = total_net[3]
    m1.metric("D360 Forecast (Net)", f"{d360_val:.2f}x", delta=f"Range: {d360_val*0.8:.2f}-{d360_val*1.2:.2f}")
    m2.metric("D180 Forecast (Net)", f"{total_net[2]:.2f}x")
    m3.metric("Implied LTV Multiplier", f"{(total_net[3]/((y_iap[-1]*GROSS_TO_NET)+y_ad[-1]) if any(y_iap+y_ad>0) else 0):.1f}x")

    # Veri Tablosu
    df_res = pd.DataFrame({
        "Day": FUTURE_DAYS,
        "IAP Forecast": iap_net.round(3),
        "Ad Forecast": ad_net.round(3),
        "NET ROAS": total_net.round(3),
        "Conservative": (total_net * 0.8).round(3),
        "Optimistic": (total_net * 1.2).round(3)
    })
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    # Grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=total_net, mode='lines+markers', name='Net Forecast', line=dict(color='#0068c9', width=4)))
    fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=iap_net, mode='lines', name='IAP Net', line=dict(dash='dash', color='#29b09d')))
    fig.update_layout(template="plotly_white", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)
