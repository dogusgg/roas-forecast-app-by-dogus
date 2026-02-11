import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Hybrid-calibrated Gompertz · Retention drives SPEED (not ceiling)")

FUTURE_DAYS = np.array([90,120,180,360,720])

# -------------------------------------------------
# Revenue
# -------------------------------------------------

fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%","85%","Custom"])

if fee_option=="70%":
    IAP_GROSS_TO_NET = 0.70
elif fee_option=="85%":
    IAP_GROSS_TO_NET = 0.85
else:
    IAP_GROSS_TO_NET = st.number_input("Custom IAP_GROSS_TO_NET",0.0,1.0,0.70,0.01)

# -------------------------------------------------
# Optional Retention Days
# -------------------------------------------------

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

for i,day in enumerate(ret_days):
    with cols[i%3]:
        ret[day] = st.number_input(
            f"D{day} Retention",
            0.0,1.0,
            0.40 if day==1 else 0.20 if day==7 else 0.10,
            0.01
        )

# fallback if optional not entered
d1 = ret.get(1,0.4)
d7 = ret.get(7,0.2)
d28 = ret.get(28,0.1)

# -------------------------------------------------
# ROAS INPUT
# -------------------------------------------------

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
        roas_iap[d] = st.number_input(f"IAP Day {d}",0.0,step=0.01)
    with c2:
        roas_ad[d] = st.number_input(f"AD Day {d}",0.0,step=0.01)

x = np.array(sorted(roas_days))
y_iap = np.array([roas_iap[d] for d in x])
y_ad = np.array([roas_ad[d] for d in x])

positive_points = np.sum(y_iap>0) + np.sum(y_ad>0)

run = st.button(
    "🚀 Generate Forecast",
    use_container_width=True,
    disabled = positive_points < 3
)

if positive_points < 3:
    st.info("Enter at least 3 positive ROAS values.")

if not run:
    st.stop()

# -------------------------------------------------
# PRODUCTION GOMPERTZ
# -------------------------------------------------

def gompertz_forecast(days, roas, is_iap=True):

    mask = roas>0
    days = days[mask]
    roas = roas[mask]

    if len(roas) < 3:
        return np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS)),np.zeros(len(FUTURE_DAYS))

    last_day = days.max()
    last_roas = roas[-1]

    # ---------------------------
    # REGIME (HYBRID calibrated)
    # ---------------------------

    regime = 3.8 if is_iap else 2.5

    # monetization signal
    if 7 in days and last_day>=28:
        growth = last_roas / max(roas[days==7][0],0.01)
        regime *= np.clip(growth/2.0,0.85,1.25)

    # HARD CLAMP → runaway killer
    ceiling = min(last_roas * regime, last_roas * 5.5)

    # ---------------------------
    # RETENTION -> SPEED ONLY
    # ---------------------------

    ret_factor = np.exp(2*(d28-0.10))   # nonlinear sensitivity

    b = 3.0 / ret_factor
    c = 120 + 220*(1-d28)

    forecast = ceiling * np.exp(-b * np.exp(-FUTURE_DAYS/c))

    # ---------------------------
    # UNCERTAINTY
    # ---------------------------

    width = np.clip(0.22 - d28*0.6, 0.08, 0.22)

    low = forecast*(1-width)
    high = forecast*(1+width)

    return forecast,low,high


iap_mean,iap_low,iap_high = gompertz_forecast(x,y_iap,True)
ad_mean,ad_low,ad_high = gompertz_forecast(x,y_ad,False)

net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
net_low = IAP_GROSS_TO_NET * iap_low + ad_low
net_high = IAP_GROSS_TO_NET * iap_high + ad_high

# -------------------------------------------------
# TABLE
# -------------------------------------------------

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

# -------------------------------------------------
# CHART
# -------------------------------------------------

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

st.plotly_chart(fig,use_container_width=True)
