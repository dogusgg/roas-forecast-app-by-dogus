import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(page_title="ROAS Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Retention-driven Hybrid · Production Style")

FUTURE_DAYS = np.array([90,120,180,360,720])

############################################################
# REVENUE PARAM
############################################################

st.subheader("Revenue Parameters")

fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%", "85%", "Custom"])

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
# RETENTION
############################################################

st.subheader("Retention Inputs")

ret1 = st.number_input("D1 retention %",0.0,100.0,40.0)/100
ret7 = st.number_input("D7 retention %",0.0,100.0,20.0)/100
ret28 = st.number_input("D28 retention %",0.0,100.0,10.0)/100

ret_days = np.array([1,7,28])
ret_vals = np.array([ret1,ret7,ret28])

def weibull(t, lam, k):
    return np.exp(-(t/lam)**k)

params,_ = curve_fit(
    weibull,
    ret_days,
    ret_vals,
    bounds=(0,[200,3])
)

lam,k = params

half_life = lam * (np.log(2))**(1/k)

st.caption(f"Estimated retention half-life ≈ **{int(half_life)} days**")

############################################################
# MULTIPLIER ENGINE
############################################################

def multiplier_from_half_life(h):

    if h < 20:
        return 1.6
    elif h < 40:
        return 2.0
    elif h < 60:
        return 2.6
    elif h < 90:
        return 3.2
    else:
        return 4.0

base_mult = multiplier_from_half_life(half_life)

############################################################
# ROAS INPUT
############################################################

st.subheader("ROAS Inputs")

roas = {
    1: st.number_input("ROAS D1",0.0,step=0.01),
    3: st.number_input("ROAS D3",0.0,step=0.01),
    7: st.number_input("ROAS D7",0.0,step=0.01),
    14: st.number_input("ROAS D14",0.0,step=0.01),
    28: st.number_input("ROAS D28",0.0,step=0.01),
}

x = np.array([d for d,v in roas.items() if v>0])
y = np.array([v for v in roas.values() if v>0])

run_enabled = len(y) >= 3

run_forecast = st.button(
    "🚀 Generate Forecast",
    disabled=not run_enabled
)

############################################################
# FORECAST
############################################################

if run_forecast:

    anchor = roas[28]

    ########################################################
    # slope boost
    ########################################################

    slope = np.polyfit(np.log(x),np.log(y),1)[0]

    slope_boost = np.clip((slope-0.35)*0.6,0,0.25)

    final_mult = base_mult * (1+slope_boost)

    st.caption(f"ROAS360 multiplier ≈ **{final_mult:.2f}x**")

    roas360 = anchor * final_mult

    ########################################################
    # curve
    ########################################################

    beta = np.clip(slope,0.25,0.6)

    def curve(t):
        raw = anchor * (t/28)**beta
        return np.minimum(raw,roas360)

    preds = curve(FUTURE_DAYS)

    ########################################################
    # confidence
    ########################################################

    low_mult = final_mult*0.85
    high_mult = final_mult*1.15

    low = np.minimum(anchor*(FUTURE_DAYS/28)**beta, anchor*low_mult)
    high = np.minimum(anchor*(FUTURE_DAYS/28)**beta, anchor*high_mult)

    ########################################################
    # NET
    ########################################################

    net_mean = preds * IAP_GROSS_TO_NET
    net_low = low * IAP_GROSS_TO_NET
    net_high = high * IAP_GROSS_TO_NET

    df = pd.DataFrame({
        "Day":FUTURE_DAYS,
        "ROAS_IAP":preds,
        "ROAS_NET":net_mean,
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

    fig.add_trace(go.Scatter(
        x=FUTURE_DAYS,
        y=preds,
        line=dict(dash="dash"),
        name="IAP"
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Observed"
    ))

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=520
    )

    fig.update_xaxes(type="log")

    st.plotly_chart(fig,use_container_width=True)
