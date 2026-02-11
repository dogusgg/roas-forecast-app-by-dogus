import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Hill Saturation Model · Retention-aware · IAP / AD separated")

FUTURE_DAYS = np.array([90,120,180,360,720])

# ------------------------
# Revenue Parameters
# ------------------------
st.subheader("Revenue Parameters")

fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%","85%","Custom"])

if fee_option=="70%":
    IAP_GROSS_TO_NET = 0.70
elif fee_option=="85%":
    IAP_GROSS_TO_NET = 0.85
else:
    IAP_GROSS_TO_NET = st.number_input("Custom IAP_GROSS_TO_NET",0.0,1.0,0.70,0.01)

# ------------------------
# Retention
# ------------------------
st.subheader("Retention Inputs")

c1,c2,c3 = st.columns(3)
with c1:
    d1 = st.number_input("D1 Retention",0.0,1.0,0.40,0.01)
with c2:
    d7 = st.number_input("D7 Retention",0.0,1.0,0.20,0.01)
with c3:
    d28 = st.number_input("D28 Retention",0.0,1.0,0.10,0.01)

ret_score = 0.4*d1 + 0.35*d7 + 0.25*d28

# ------------------------
# ROAS Inputs
# ------------------------
st.subheader("ROAS Inputs")

days = np.array([1,3,7,14,28])
roas_iap = {}
roas_ad = {}

for d in days:
    c1,c2 = st.columns(2)
    with c1:
        roas_iap[d] = st.number_input(f"ROAS_IAP Day {d}",min_value=0.0,step=0.01)
    with c2:
        roas_ad[d] = st.number_input(f"ROAS_AD Day {d}",min_value=0.0,step=0.01)

y_iap = np.array([roas_iap[d] for d in days])
y_ad = np.array([roas_ad[d] for d in days])

positive_points = np.sum(y_iap>0) + np.sum(y_ad>0)

run_forecast = st.button(
    "🚀 Generate Forecast",
    type="primary",
    use_container_width=True,
    disabled = positive_points < 3
)

if positive_points < 3:
    st.info("Enter at least 3 positive ROAS values.")
if not run_forecast:
    st.stop()

# ------------------------
# Hill Model
# ------------------------

def hill_forecast(x,y):

    mask = y>0
    x = x[mask]
    y = y[mask]

    if len(y)<3:
        return np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS))

    roas_last = y[-1]

    # retention-based multiplier
    multiplier = 1.8 + 6*ret_score
    L = roas_last * multiplier

    # slope for shape
    beta = np.polyfit(np.log(x),np.log(y),1)[0]
    h = np.clip(beta,0.4,1.2)

    # half-saturation day
    k = 150 + 200*(ret_score)

    forecast = L * (FUTURE_DAYS**h) / (k**h + FUTURE_DAYS**h)

    # confidence
    width = max(0.05,0.18-0.6*ret_score)
    low = forecast*(1-width)
    high = forecast*(1+width)

    return forecast,low,high

iap_mean,iap_low,iap_high = hill_forecast(days,y_iap)
ad_mean,ad_low,ad_high = hill_forecast(days,y_ad)

net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
net_low = IAP_GROSS_TO_NET * iap_low + ad_low
net_high = IAP_GROSS_TO_NET * iap_high + ad_high

# ------------------------
# Output Table
# ------------------------
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

# ------------------------
# Plot
# ------------------------
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
    x=days[mask_obs],
    y=y_iap[mask_obs],
    mode="markers",
    name="Observed IAP"
))

fig.update_layout(
    template="plotly_white",
    height=520,
    hovermode="x unified"
)

st.plotly_chart(fig,use_container_width=True)
