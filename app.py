import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Deterministic Hill Saturation · Precision Calibrated")

FUTURE_DAYS = np.array([90,120,180,360,720])

# --- Revenue Parameters ---
st.subheader("Revenue Parameters")
fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%","85%","Custom"])
IAP_GROSS_TO_NET = 0.70 if fee_option=="70%" else (0.85 if fee_option=="85%" else st.number_input("Custom Value", 0.0, 1.0, 0.70))

# --- Retention Inputs ---
st.subheader("Retention Inputs")
ret_days_default = [1,7,28]
ret_days = st.multiselect("Select retention days", [1,3,7,14,28,45,60], default=ret_days_default)
ret = {}
cols = st.columns(3)
for i, d in enumerate(sorted(ret_days)):
    with cols[i % 3]:
        default_val = [0.40,0.20,0.10][ret_days_default.index(d)] if d in ret_days_default else 0.0
        ret[d] = st.number_input(f"D{d} Retention", 0.0, 1.0, default_val, 0.01)

def retention_quality(ret):
    d1, d7, d28 = ret.get(1,0), ret.get(7, 0.2), ret.get(28, 0.1)
    return np.clip((0.55*d28 + 0.30*d7 + 0.15*d1), 0.05, 0.6)

ret_q = retention_quality(ret)

# --- ROAS Inputs ---
st.subheader("ROAS Inputs")
roas_days_default = [1,3,7,14,28]
roas_days = st.multiselect("Select ROAS days", [1,3,7,14,28,45,60], default=roas_days_default)
roas_iap, roas_ad = {}, {}

for d in sorted(roas_days):
    c1, c2 = st.columns(2)
    with c1: roas_iap[d] = st.number_input(f"ROAS_IAP Day {d}", 0.0, step=0.01)
    with c2: roas_ad[d] = st.number_input(f"ROAS_AD Day {d}", 0.0, step=0.01)

x = np.array(sorted(roas_days))
y_iap = np.array([roas_iap[d] for d in x])
y_ad = np.array([roas_ad[d] for d in x])

# BUTON KILIDI (KESİN 3 NOKTA KOŞULU)
total_positive_iap = np.sum(y_iap > 0)
total_positive_ad = np.sum(y_ad > 0)
total_points = total_positive_iap + total_positive_ad

run = st.button("🚀 Generate Forecast", use_container_width=True, type="primary", disabled=total_points < 3)

if not run:
    if total_points < 3:
        st.info("⚠️ Tahmin için toplamda en az 3 adet (IAP veya AD) pozitif ROAS değeri girilmelidir.")
    st.stop()

####################################################
# 🔥 STABLE HILL MODEL (CALIBRATED)
####################################################

def stable_hill_forecast(x_all, y_all, ret_q):
    mask = y_all > 0
    # Eğer bu kanal için hiç veri yoksa (0'sa), crash olmaması için 0 dön
    if np.sum(mask) == 0:
        return np.zeros(len(FUTURE_DAYS)), np.zeros(len(FUTURE_DAYS)), np.zeros(len(FUTURE_DAYS))
    
    # En az 1 veri varsa ama 3'ten azsa, lineer bir baseline oluştur (crash engelleme)
    x_f, y_f = x_all[mask], y_all[mask]
    last_d, last_r = x_f[-1], y_f[-1]
    first_r = y_f[0]
    
    # Growth scaling
    raw_growth = last_r / max(first_r, 0.01)
    time_weight = (28 / last_d)**0.7
    growth_factor = np.clip(raw_growth * time_weight, 1.1, 4.5)
    
    # Multiplier (Senaryolarına göre ayarlandı)
    ltv_mult = 4.5 + (9.0 * ret_q) + (1.2 * (growth_factor - 1))
    ltv_mult = np.clip(ltv_mult, 3.0, 12.0)
    
    ceiling = last_r * ltv_mult * (28 / last_d)**0.05
    h = np.clip(0.85 + 0.6 * ret_q, 0.9, 1.4)
    k = 180 + 340 * (1 - ret_q)
    
    forecast = ceiling * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)
    width = np.clip(0.18 - 0.22 * ret_q, 0.06, 0.16)
    return forecast, forecast*(1-width), forecast*(1+width)

# Hesaplamalar
iap_m, iap_l, iap_h = stable_hill_forecast(x, y_iap, ret_q)
ad_m, ad_l, ad_h = stable_hill_forecast(x, y_ad, ret_q)

net_m = (IAP_GROSS_TO_NET * iap_m) + ad_m
net_l = (IAP_GROSS_TO_NET * iap_l) + ad_l
net_h = (IAP_GROSS_TO_NET * iap_h) + ad_h

####################################################
# RESULTS
####################################################
st.subheader("Forecast Results")
df = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "ROAS_IAP": iap_m.round(3),
    "ROAS_AD": ad_m.round(3),
    "ROAS_NET": net_m.round(3),
    "NET_low": net_l.round(3),
    "NET_high": net_h.round(3)
})
st.dataframe(df, hide_index=True, use_container_width=True)

st.subheader("ROAS Curves")
fig = go.Figure()

# Confidence Band
fig.add_trace(go.Scatter(
    x=np.concatenate([FUTURE_DAYS, FUTURE_DAYS[::-1]]),
    y=np.concatenate([net_h, net_l[::-1]]),
    fill="toself", fillcolor="rgba(150,150,150,0.25)",
    line=dict(color="rgba(255,255,255,0)"), name="Confidence Band"
))

# Mean Lines
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=net_m, mode="lines+markers", name="NET Forecast", line=dict(width=4, color="blue")))
fig.add_trace(go.Scatter(x=FUTURE_DAYS, y=iap_m, mode="lines", name="IAP Forecast", line=dict(dash="dash", color="green")))

# Observed Data
if total_positive_iap > 0:
    fig.add_trace(go.Scatter(x=x[y_iap > 0], y=y_iap[y_iap > 0], mode="markers", name="Observed IAP", marker=dict(color="green", size=10)))

fig.update_layout(template="plotly_white", height=520, hovermode="x unified", xaxis_title="Days", yaxis_title="ROAS")
st.plotly_chart(fig, use_container_width=True)
