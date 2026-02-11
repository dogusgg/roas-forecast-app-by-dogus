import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Deterministic Hill Saturation · Retention-aware · IAP / AD separated")

FUTURE_DAYS = np.array([90,120,180,360,720])

####################################################
# Revenue
####################################################

st.subheader("Revenue Parameters")

fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%","85%","Custom"])

if fee_option=="70%":
    IAP_GROSS_TO_NET = 0.70
elif fee_option=="85%":
    IAP_GROSS_TO_NET = 0.85
else:
    IAP_GROSS_TO_NET = st.number_input(
        "Custom IAP_GROSS_TO_NET",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01
    )

####################################################
# RETENTION (optional days supported)
####################################################

st.subheader("Retention Inputs")

ret_days_default = [1,7,28]
ret_days_optional = [3,14,45,60]

ret_days = st.multiselect(
    "Select retention days",
    options=ret_days_default + ret_days_optional,
    default=ret_days_default
)

ret = {}

cols = st.columns(3)

for i, d in enumerate(sorted(ret_days)):
    with cols[i % 3]:
        ret[d] = st.number_input(
            f"D{d} Retention",
            min_value=0.0,
            max_value=1.0,
            value=0.0 if d not in ret_days_default else [0.40,0.20,0.10][ret_days_default.index(d)],
            step=0.01
        )

def retention_quality(ret):
    # fallback logic
    d1 = ret.get(1,0)
    d7 = ret.get(7, d1*0.5)
    d28 = ret.get(28, d7*0.5)

    # strong weighting → sensitivity artırıldı
    ret_q = (
        0.55*d28 + 
        0.30*d7 + 
        0.15*d1
    )

    return np.clip(ret_q,0.05,0.6)

ret_q = retention_quality(ret)

####################################################
# ROAS INPUT
####################################################

st.subheader("ROAS Inputs")

roas_days_default = [1,3,7,14,28]
roas_days_optional = [45,60]

roas_days = st.multiselect(
    "Select ROAS days",
    options=roas_days_default + roas_days_optional,
    default=roas_days_default
)

roas_iap = {}
roas_ad = {}

for d in sorted(roas_days):
    c1,c2 = st.columns(2)

    with c1:
        roas_iap[d] = st.number_input(f"ROAS_IAP Day {d}",0.0,step=0.01)

    with c2:
        roas_ad[d] = st.number_input(f"ROAS_AD Day {d}",0.0,step=0.01)

x = np.array(sorted(roas_days))
y_iap = np.array([roas_iap[d] for d in x])
y_ad = np.array([roas_ad[d] for d in x])

positive_points = np.sum(y_iap>0) + np.sum(y_ad>0)

run = st.button(
    "🚀 Generate Forecast",
    use_container_width=True,
    type="primary",
    disabled=positive_points < 3
)

if positive_points < 3:
    st.info("Enter at least 3 positive ROAS values.")

if not run:
    st.stop()

####################################################
# 🔥 STABLE HILL MODEL
####################################################

def stable_hill_forecast(x,y,ret_q):

    mask = y>0
    x_f = x[mask]
    y_f = y[mask]

    if len(y_f) < 3:
        return np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS))

    # anchor
    last_d, last_r = x_f[-1], y_f[-1]
    first_r = y_f[0]

    # growth scaling
    raw_growth = last_r / max(first_r, 0.01)
    time_correction = (28 / last_d)**0.5
    growth_factor = np.clip(raw_growth * time_correction, 1.5, 8.0)

    # 🔥 LTV MULTIPLIER (BEKLENTİLERİNE GÖRE TUNED)
    ltv_mult = 4.2 + 13.0 * ret_q + 2.8 * (growth_factor - 1)
    ltv_mult = np.clip(ltv_mult, 3.5, 25.0)

    ceiling = last_r * ltv_mult * (28 / last_d)**0.2

    # Shape params
    h = np.clip(0.80 + 0.85 * ret_q, 0.85, 1.6)
    k = 180 + 350 * (1 - ret_q)

    forecast = ceiling * (FUTURE_DAYS**h)/(k**h + FUTURE_DAYS**h)

    # Confidence
    width = np.clip(0.18 - 0.22*ret_q, 0.06, 0.16)
    low = forecast*(1-width)
    high = forecast*(1+width)

    return forecast,low,high


iap_mean,iap_low,iap_high = stable_hill_forecast(x,y_iap,ret_q)
ad_mean,ad_low,ad_high = stable_hill_forecast(x,y_ad,ret_q)

net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
net_low = IAP_GROSS_TO_NET * iap_low + ad_low
net_high = IAP_GROSS_TO_NET * iap_high + ad_high

####################################################
# TABLE
####################################################

st.subheader("Forecast")

df = pd.DataFrame({
    "Day":FUTURE_DAYS,
    "ROAS_IAP":np.round(iap_mean,3),
    "ROAS_AD":np.round(ad_mean,3),
    "ROAS_NET":np.round(net_mean,3),
    "NET_low":np.round(net_low,3),
    "NET_high":np.round(net_high,3)
})

st.dataframe(df,hide_index=True,use_container_width=True)

####################################################
# PLOT
####################################################

st.subheader("ROAS Curves")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.concatenate([FUTURE_DAYS,FUTURE_DAYS[::-1]]),
    y=np.concatenate([net_high,net_low[::-1]]),
    fill="toself",
    fillcolor="rgba(150,150,150,0.25)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence"
))

fig.add_trace(go.Scatter(
    x=FUTURE_DAYS,
    y=net_mean,
    mode="lines",
    line=dict(width=4),
    name="NET"
))

fig.add_trace(go.Scatter(
    x=FUTURE_DAYS,
    y=iap_mean,
    mode="lines",
    line=dict(dash="dash"),
    name="IAP"
))

mask_obs = y_iap>0
fig.add_trace(go.Scatter(
    x=x[mask_obs],
    y=y_iap[mask_obs],
    mode="markers",
    name="Observed IAP"
))

fig.update_layout(
    template="plotly_white",
    height=520,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
