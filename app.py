import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Gompertz Growth · Retention-weighted · IAP / AD separated")

FUTURE_DAYS = np.array([90,120,180,360,720])

# -------------------
# Revenue
# -------------------

fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%","85%","Custom"])

if fee_option=="70%":
    IAP_GROSS_TO_NET = 0.70
elif fee_option=="85%":
    IAP_GROSS_TO_NET = 0.85
else:
    IAP_GROSS_TO_NET = st.number_input("Custom",0.0,1.0,0.70,0.01)

# -------------------
# Retention
# -------------------

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

# weighted retention score (nonlinear)
ret_values = np.array(list(ret.values()))

if len(ret_values)>0:
    retention_score = np.mean(np.sqrt(ret_values))  # <-- key trick
else:
    retention_score = 0.1

# -------------------
# ROAS Inputs
# -------------------

st.subheader("ROAS Inputs")

roas_days_default = [1,3,7,14,28]
roas_days_optional = [45,60]

roas_days = st.multiselect(
    "Select ROAS days",
    options=roas_days_default + roas_days_optional,
    default=roas_days_default
)

roas_iap={}
roas_ad={}

for d in roas_days:
    c1,c2 = st.columns(2)

    with c1:
        roas_iap[d] = st.number_input(f"IAP Day {d}",0.0,10.0,0.0,0.01)

    with c2:
        roas_ad[d] = st.number_input(f"AD Day {d}",0.0,10.0,0.0,0.01)

# observed arrays
x = np.array(roas_days)

y_iap = np.array([roas_iap[d] for d in roas_days])
y_ad = np.array([roas_ad[d] for d in roas_days])

positive_points = np.sum(y_iap>0) + np.sum(y_ad>0)

run = st.button(
    "🚀 Generate Forecast",
    type="primary",
    use_container_width=True,
    disabled = positive_points < 3
)

if positive_points < 3:
    st.info("Enter at least 3 positive ROAS values.")

if not run:
    st.stop()

# -------------------
# GOMPERTZ MODEL
# -------------------

def gompertz_forecast(x,y,retention_score):

    mask = y>0
    x=x[mask]
    y=y[mask]

    if len(y)<3:
        return np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS))

    # anchor
    last_roas = y[-1]

    # detect early spike
    if 7 in x:
        r7 = y[x==7][0]
    else:
        r7 = y[1]

    growth = last_roas / max(r7,0.01)

    # INDUSTRY LTV MULTIPLIER
    base_mult = 3.5 + 9*retention_score

    # spike bonus
    spike_bonus = np.clip(growth,1.5,4)

    L = last_roas * base_mult * spike_bonus

    # gompertz params
    b = 3.2 - retention_score*2
    c = 0.015 + (1-retention_score)*0.01

    forecast = L * np.exp(-b*np.exp(-c*FUTURE_DAYS))

    # uncertainty
    width = 0.18 - retention_score*0.08
    low = forecast*(1-width)
    high = forecast*(1+width)

    return forecast,low,high


iap_mean,iap_low,iap_high = gompertz_forecast(x,y_iap,retention_score)
ad_mean,ad_low,ad_high = gompertz_forecast(x,y_ad,retention_score)

net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
net_low = IAP_GROSS_TO_NET * iap_low + ad_low
net_high = IAP_GROSS_TO_NET * iap_high + ad_high

# -------------------
# TABLE
# -------------------

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

# -------------------
# PLOT
# -------------------

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

st.plotly_chart(fig,use_container_width=True)

