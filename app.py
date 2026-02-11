import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Deterministic Hill Saturation · Optimized for Early Signals")

FUTURE_DAYS = np.array([90,120,180,360,720])

# --- Revenue & Fee ---
st.subheader("Revenue Parameters")
fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%","85%","Custom"])
IAP_GROSS_TO_NET = 0.70 if fee_option=="70%" else (0.85 if fee_option=="85%" else st.number_input("Custom Value", 0.0, 1.0, 0.70))

# --- Retention ---
st.subheader("Retention Inputs")
ret_days = st.multiselect("Select retention days", [1,3,7,14,28,45,60], default=[1,7,28])
ret = {}
cols = st.columns(3)
for i, d in enumerate(sorted(ret_days)):
    with cols[i % 3]:
        default_ret = [0.40, 0.20, 0.10][ [1,7,28].index(d) ] if d in [1,7,28] else 0.0
        ret[d] = st.number_input(f"D{d} Retention", 0.0, 1.0, default_ret, step=0.01)

def retention_quality(ret):
    d1, d7, d28 = ret.get(1,0), ret.get(7, 0.2), ret.get(28, 0.1)
    # Retention sensitivity artırıldı
    ret_q = (0.55*d28 + 0.30*d7 + 0.15*d1)
    return np.clip(ret_q, 0.05, 0.6)

ret_q = retention_quality(ret)

# --- ROAS Input ---
st.subheader("ROAS Inputs")
roas_days = st.multiselect("Select ROAS days", [1,3,7,14,28,45,60], default=[1,3,7,14,28])
roas_iap, roas_ad = {}, {}
for d in sorted(roas_days):
    c1, c2 = st.columns(2)
    with c1: roas_iap[d] = st.number_input(f"ROAS_IAP Day {d}", 0.0, step=0.01)
    with c2: roas_ad[d] = st.number_input(f"ROAS_AD Day {d}", 0.0, step=0.01)

x = np.array(sorted(roas_days))
y_iap, y_ad = np.array([roas_iap[d] for d in x]), np.array([roas_ad[d] for d in x])

if st.button("🚀 Generate Forecast", use_container_width=True, type="primary"):
    
    def stable_hill_forecast(x_all, y_all, ret_q):
        mask = y_all > 0
        if np.sum(mask) < 2: return np.zeros(5), np.zeros(5), np.zeros(5)
        x_f, y_f = x_all[mask], y_all[mask]
        
        last_d, last_r = x_f[-1], y_f[-1]
        first_r = y_f[0]
        
        # Growth & Scaling (Simülasyona göre tune edildi)
        raw_growth = last_r / max(first_r, 0.01)
        time_correction = (28 / last_d)**0.5
        growth_factor = np.clip(raw_growth * time_correction, 1.5, 8.0)
        
        # LTV Multiplier (Vaka 1-2-3 fixleri burada)
        ltv_mult = 4.2 + (13.0 * ret_q) + (2.8 * (growth_factor - 1))
        ltv_mult = np.clip(ltv_mult, 3.5, 25.0)
        
        ceiling = last_r * ltv_mult * (28 / last_d)**0.2
        h = np.clip(0.8 + 0.85 * ret_q, 0.85, 1.6)
        k = 180 + 350 * (1 - ret_q)
        
        forecast = ceiling * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)
        width = np.clip(0.18 - 0.2*ret_q, 0.06, 0.16)
        return forecast, forecast*(1-width), forecast*(1+width)

    iap_m, iap_l, iap_h = stable_hill_forecast(x, y_iap, ret_q)
    ad_m, ad_l, ad_h = stable_hill_forecast(x, y_ad, ret_q)
    
    net_m = (IAP_GROSS_TO_NET * iap_m) + ad_m
    
    st.subheader("Forecast Results")
    res_df = pd.DataFrame({"Day": FUTURE_DAYS, "IAP": iap_m.round(3), "AD": ad_m.round(3), "NET": net_m.round(3)})
    st.dataframe(res_df, hide_index=True, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_m, mode="lines+markers", name="NET Forecast", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=x[y_iap>0], y=y_iap[y_iap>0], mode="markers", name="Observed IAP"))
    fig.update_layout(template="plotly_white", hovermode="x unified", xaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
