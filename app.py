import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(layout="centered")

st.title("📈 ROAS Forecast Engine")
st.caption("Industry-style IAP + AD modeling")

FUTURE = np.array([90,120,180,360,720])

########################################################
# INPUT
########################################################

st.subheader("IAP ROAS")

iap_data = {
    1: st.number_input("IAP D1",0.0,step=0.01),
    3: st.number_input("IAP D3",0.0,step=0.01),
    7: st.number_input("IAP D7",0.0,step=0.01),
    14: st.number_input("IAP D14",0.0,step=0.01),
    28: st.number_input("IAP D28",0.0,step=0.01),
}

st.subheader("AD ROAS")

ad_data = {
    1: st.number_input("AD D1",0.0,step=0.01),
    3: st.number_input("AD D3",0.0,step=0.01),
    7: st.number_input("AD D7",0.0,step=0.01),
    14: st.number_input("AD D14",0.0,step=0.01),
    28: st.number_input("AD D28",0.0,step=0.01),
}

########################################################
# CURVES
########################################################

def gompertz(t,a,b,c):
    return a*np.exp(-b*np.exp(-c*t))

def fit_iap(days,values):

    x = np.array(days)
    y = np.array(values)

    if len(y) < 3:
        return np.zeros(len(FUTURE))

    try:
        popt,_ = curve_fit(
            gompertz,
            x,y,
            bounds=(0,[5,10,1]),
            maxfev=10000
        )

        preds = gompertz(FUTURE,*popt)

        # runaway guard
        cap = y[-1]*4
        return np.minimum(preds,cap)

    except:
        slope = np.polyfit(np.log(x),y,1)[0]
        return y[-1] + slope*np.log(FUTURE/x[-1])

def fit_ad(days,values):

    x = np.array(days)
    y = np.array(values)

    if len(y) < 3:
        return np.zeros(len(FUTURE))

    slope = np.polyfit(np.log(x),y,1)[0]

    preds = y[-1] + slope*np.log(FUTURE/x[-1])

    # ads rarely explode
    cap = y[-1]*2.5
    return np.minimum(preds,cap)

########################################################
# PREP
########################################################

iap_days = [d for d,v in iap_data.items() if v>0]
iap_vals = [v for v in iap_data.values() if v>0]

ad_days = [d for d,v in ad_data.items() if v>0]
ad_vals = [v for v in ad_data.values() if v>0]

run = st.button("🚀 Generate Forecast")

########################################################
# FORECAST
########################################################

if run:

    iap_pred = fit_iap(iap_days,iap_vals)
    ad_pred = fit_ad(ad_days,ad_vals)

    net = iap_pred + ad_pred

    df = pd.DataFrame({
        "Day":FUTURE,
        "IAP":iap_pred,
        "AD":ad_pred,
        "NET":net
    })

    st.subheader("Forecast")

    st.dataframe(
        df.round(2),
        hide_index=True,
        use_container_width=True
    )

########################################################
# GRAPH
########################################################

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=FUTURE,
        y=iap_pred,
        name="IAP",
        line=dict(dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=FUTURE,
        y=ad_pred,
        name="AD",
        line=dict(dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=FUTURE,
        y=net,
        name="NET",
        line=dict(width=4)
    ))

    fig.add_trace(go.Scatter(
        x=iap_days,
        y=iap_vals,
        mode="markers",
        name="Observed IAP"
    ))

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig,use_container_width=True)
