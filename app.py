import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Hill Saturation · Retention-aware · IAP / AD separated")

FUTURE_DAYS = np.array([90,120,180,360,720])

########################################
# Revenue
########################################

st.subheader("Revenue Parameters")

fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%","85%","Custom"])

if fee_option=="70%":
    IAP_GROSS_TO_NET = 0.70
elif fee_option=="85%":
    IAP_GROSS_TO_NET = 0.85
else:
    IAP_GROSS_TO_NET = st.number_input("Custom IAP_GROSS_TO_NET",0.0,1.0,0.70,0.01)

########################################
# Retention (45 & 60 optional)
########################################

st.subheader("Retention Inputs")

ret_days_default = [1,7,28]
ret_days_optional = [3,14,45,60]

ret_selected = st.multiselect(
    "Select retention days",
    ret_days_default + ret_days_optional,
    default=ret_days_default
)

ret = {}

cols = st.columns(len(ret_selected))

for i,d in enumerate(sorted(ret_selected)):
    with cols[i]:
        ret[d] = st.number_input(f"D{d}",0.0,1.0,0.0,0.01)

def retention_score(ret):

    if len(ret)==0:
        return 0.15

    score = 0

    weights = {
    1:0.10,
    3:0.15,
    7:0.20,
    14:0.15,
    28:0.20,
    45:0.10,
    60:0.10
    }

    for d,val in ret.items():
        score += weights.get(d,0)*val

    return np.clip(score,0.05,0.6)

ret_score = retention_score(ret)

########################################
# ROAS INPUTS (45-60 optional)
########################################

st.subheader("ROAS Inputs")

day_options = [1,3,7,14,28,45,60]

selected_days = st.multiselect(
    "Select ROAS days",
    day_options,
    default=[1,3,7,14,28]
)

days = np.array(sorted(selected_days))

roas_iap = {}
roas_ad = {}

for d in days:
    c1,c2 = st.columns(2)
    with c1:
        roas_iap[d] = st.number_input(f"ROAS_IAP Day {d}",0.0,step=0.01)
    with c2:
        roas_ad[d] = st.number_input(f"ROAS_AD Day {d}",0.0,step=0.01)

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

########################################
# STABLE HILL MODEL
########################################

def hill_forecast(x,y,ret_score):

    mask = y>0
    x = x[mask]
    y = y[mask]

    if len(y)<3:
        return np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS))

    last_day = x.max()
    roas_last = y[x.argmax()]

    # runaway protection if only early data exists
    if last_day <= 7:
        ceiling_mult = 2.5 + 3*ret_score
    elif last_day <= 14:
        ceiling_mult = 3 + 4*ret_score
    else:
        ceiling_mult = 3.5 + 6*ret_score

    L = roas_last * ceiling_mult

    # curvature from log slope
    beta = np.polyfit(np.log(x),np.log(y),1)[0]
    h = np.clip(beta,0.65,1.25)

    # retention strongly controls half-saturation
    k = 180 - (ret_score*140)

    forecast = L * (FUTURE_DAYS**h)/(k**h + FUTURE_DAYS**h)

    width = 0.18 - ret_score*0.12
    width = np.clip(width,0.06,0.18)

    low = forecast*(1-width)
    high = forecast*(1+width)

    return forecast,low,high


iap_mean,iap_low,iap_high = hill_forecast(days,y_iap,ret_score)
ad_mean,ad_low,ad_high = hill_forecast(days,y_ad,ret_score)

net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
net_low = IAP_GROSS_TO_NET * iap_low + ad_low
net_high = IAP_GROSS_TO_NET * iap_high + ad_high

########################################
# TABLE
########################################

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

########################################
# PLOT
########################################

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

# observed only >0
mask_obs = y_iap>0
fig.add_trace(go.Scatter(
    x=days[mask_obs],
    y=y_iap[mask_obs],
    mode="markers",
    name="Observed IAP"
))

mask_obs_ad = y_ad>0
fig.add_trace(go.Scatter(
    x=days[mask_obs_ad],
    y=y_ad[mask_obs_ad],
    mode="markers",
    name="Observed AD"
))

fig.update_layout(
    template="plotly_white",
    height=520,
    hovermode="x unified"
)

st.plotly_chart(fig,use_container_width=True)

