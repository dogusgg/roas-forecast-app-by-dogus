import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(page_title="ROAS Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Deterministic Power Fit · Production-grade")

FUTURE_DAYS = np.array([90,120,180,360,720])

############################################################
# REVENUE PARAM
############################################################

st.subheader("Revenue Parameters")

fee_option = st.selectbox(
    "IAP_GROSS_TO_NET",
    ["70%", "85%", "Custom"]
)

if fee_option == "70%":
    IAP_GROSS_TO_NET = 0.70
elif fee_option == "85%":
    IAP_GROSS_TO_NET = 0.85
else:
    IAP_GROSS_TO_NET = st.number_input(
        "Custom IAP_GROSS_TO_NET",
        0.0,1.0,0.70,0.01
    )

############################################################
# RETENTION → CAP INFORMER
############################################################

st.subheader("Retention (optional but recommended)")

ret_days = st.multiselect(
    "Retention days",
    [1,7,14,28,45,60],
    default=[1,7,28]
)

ret = {}

for d in ret_days:
    ret[d] = st.number_input(
        f"D{d} retention %",
        0.0,100.0,
        value=40.0 if d==1 else 20.0 if d==7 else 10.0,
        step=0.1
    ) / 100


def weibull(t, lam, k):
    return np.exp(-(t/lam)**k)

cap_multiplier = 2.5  # safe default

if len(ret_days) >= 3:
    try:
        params,_ = curve_fit(
            weibull,
            np.array(list(ret.keys())),
            np.array(list(ret.values())),
            bounds=(0,[200,3])
        )

        lam,k = params

        half_life = lam * (np.log(2))**(1/k)

        cap_multiplier = np.clip(
            1.8 + half_life/90,
            1.8,
            4.5
        )

    except:
        st.warning("Retention fit failed — using default cap.")

############################################################
# ROAS INPUT
############################################################

st.subheader("ROAS Inputs")

options = list(range(1,29)) + [45,60]

days_selected = st.multiselect(
    "ROAS days",
    options,
    default=[1,3,7,14,28]
)

roas_iap, roas_ad = {},{}

for d in sorted(days_selected):
    c1,c2 = st.columns(2)

    with c1:
        roas_iap[d] = st.number_input(f"IAP D{d}",0.0,step=0.01)

    with c2:
        roas_ad[d] = st.number_input(f"AD D{d}",0.0,step=0.01)

x = np.array(sorted(days_selected))

y_iap = np.array([roas_iap[d] for d in x])
y_ad = np.array([roas_ad[d] for d in x])

iap_pos = np.sum(y_iap>0)
ad_pos = np.sum(y_ad>0)

run_enabled = (iap_pos>=3) or (ad_pos>=3)

button_label = "🚀 Generate Forecast"
if not run_enabled:
    button_label += " (need ≥3 positive ROAS)"

run_forecast = st.button(button_label, disabled=not run_enabled)

############################################################
# POWER + SOFT CAP MODEL
############################################################

def power_softcap_forecast(x,y,cap_mult):

    mask = y>0
    x = x[mask]
    y = y[mask]

    logx = np.log(x)
    logy = np.log(y)

    slope,intercept = np.polyfit(logx,logy,1)

    a = np.exp(intercept)
    b = slope

    anchor = y[-1]

    cap = anchor * cap_mult

    def softcap(day):
        raw = a * day**b
        return cap * (1 - np.exp(-raw/cap))

    preds = softcap(FUTURE_DAYS)

    ########################################################
    # CONFIDENCE via slope variance
    ########################################################

    residuals = logy - (intercept + slope*logx)
    sigma = np.std(residuals)

    upper_slope = b + sigma
    lower_slope = max(0.05, b - sigma)

    def band(day, slope):
        raw = a * day**slope
        return cap * (1 - np.exp(-raw/cap))

    high = band(FUTURE_DAYS, upper_slope)
    low = band(FUTURE_DAYS, lower_slope)

    return preds, low, high


############################################################
# RUN
############################################################

if run_forecast:

    streams=[]

    if iap_pos>=3:
        iap_mean,iap_low,iap_high = power_softcap_forecast(
            x,y_iap,cap_multiplier
        )
        streams.append("IAP")
    else:
        iap_mean = iap_low = iap_high = np.zeros(len(FUTURE_DAYS))

    if ad_pos>=3:
        ad_mean,ad_low,ad_high = power_softcap_forecast(
            x,y_ad,1.8
        )
        streams.append("AD")
    else:
        ad_mean = ad_low = ad_high = np.zeros(len(FUTURE_DAYS))

    st.caption(f"Forecast generated for: {', '.join(streams)}")

    net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
    net_low = IAP_GROSS_TO_NET * iap_low + ad_low
    net_high = IAP_GROSS_TO_NET * iap_high + ad_high

    df = pd.DataFrame({
        "Day":FUTURE_DAYS,
        "IAP":iap_mean,
        "AD":ad_mean,
        "NET":net_mean,
        "NET_low":net_low,
        "NET_high":net_high
    })

    st.subheader("Forecast")

    st.dataframe(
        df.round(2),
        hide_index=True,
        use_container_width=True
    )

    ########################################################
    # PAYBACK
    ########################################################

    if net_mean.max() >= 1:
        payback = np.interp(1,net_mean,FUTURE_DAYS)
        st.success(f"✅ Expected Payback ≈ Day {int(payback)}")
    else:
        st.warning("⚠️ Payback not reached within 720 days")

    ########################################################
    # GRAPH
    ########################################################

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([FUTURE_DAYS,FUTURE_DAYS[::-1]]),
        y=np.concatenate([net_high,net_low[::-1]]),
        fill="toself",
        fillcolor="rgba(150,150,150,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence"
    ))

    fig.add_trace(go.Scatter(
        x=FUTURE_DAYS,
        y=net_mean,
        line=dict(width=4),
        name="NET"
    ))

    if iap_pos>=3:
        fig.add_trace(go.Scatter(
            x=FUTURE_DAYS,
            y=iap_mean,
            line=dict(dash="dash"),
            name="IAP"
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=y_iap,
            mode="markers",
            name="Observed IAP"
        ))

    if ad_pos>=3:
        fig.add_trace(go.Scatter(
            x=FUTURE_DAYS,
            y=ad_mean,
            line=dict(dash="dot"),
            name="AD"
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=y_ad,
            mode="markers",
            name="Observed AD"
        ))

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=520
    )

    fig.update_xaxes(type="log")

    st.plotly_chart(fig,use_container_width=True)
