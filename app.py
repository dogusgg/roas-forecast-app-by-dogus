import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
from scipy.optimize import curve_fit

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Bayesian power-law + Weibull retention")

FUTURE_DAYS = np.array([90,120,180,360,720])

############################################
# Revenue params
############################################

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

############################################
# RETENTION INPUT
############################################

st.subheader("Retention (Curve Fit)")

ret_days = st.multiselect(
    "Retention days",
    [1,3,7,14,30,45,60],
    default=[1,7,30]
)

ret_values = {}

for d in sorted(ret_days):
    ret_values[d] = st.number_input(
        f"D{d} retention %",
        0.0,100.0,
        value=35.0 if d==1 else 12.0 if d==7 else 5.0,
        step=0.1
    ) / 100


############################################
# Weibull Fit
############################################

def weibull(t, lam, k):
    return np.exp(-(t/lam)**k)

def fit_weibull(days, vals):
    days = np.array(days)
    vals = np.array(vals)

    params,_ = curve_fit(
        weibull,
        days,
        vals,
        bounds=(0,[200,3])
    )

    lam,k = params
    return lam,k


if len(ret_days) >= 3:
    lam,k = fit_weibull(list(ret_values.keys()),
                       list(ret_values.values()))

    retention_alpha = -k / lam**k
else:
    retention_alpha = None


############################################
# ROAS INPUT
############################################

st.subheader("ROAS Inputs")

options = list(range(1,29)) + [45,60]

days_selected = st.multiselect(
    "ROAS days",
    options,
    default=[1,3,7,14,28]
)

if len(days_selected) < 3:
    st.stop()

roas_iap, roas_ad = {},{}

for d in sorted(days_selected):
    c1,c2 = st.columns(2)

    with c1:
        roas_iap[d] = st.number_input(
            f"IAP D{d}",0.0,step=0.01)

    with c2:
        roas_ad[d] = st.number_input(
            f"AD D{d}",0.0,step=0.01)

x = np.array(sorted(days_selected))
y_iap = np.array([roas_iap[d] for d in x])
y_ad = np.array([roas_ad[d] for d in x])

iap_pos = np.sum(y_iap>0)
ad_pos = np.sum(y_ad>0)

run_enabled = (iap_pos>=3) or (ad_pos>=3)

if not run_enabled:
    st.stop()

############################################
# Bayesian Power Law
############################################

@st.cache_resource
def bayesian_power(x,y,ret_alpha):

    mask = y>0
    x = x[mask]
    y = y[mask]

    log_x = np.log(x)
    log_y = np.log(y)

    anchor_day = max(x)
    anchor = y[list(x).index(anchor_day)]

    roas_alpha = np.polyfit(log_x,log_y,1)[0]

    if ret_alpha is not None:
        alpha_prior = 0.55*roas_alpha + 0.45*ret_alpha
    else:
        alpha_prior = roas_alpha

    with pm.Model() as model:

        alpha = pm.Normal(
            "alpha",
            mu=alpha_prior,
            sigma=0.05
        )

        c = pm.Normal(
            "c",
            mu=np.log(anchor),
            sigma=0.25
        )

        sigma = pm.HalfNormal(
            "sigma",
            0.06
        )

        mu = c + alpha*log_x

        pm.Normal("obs",mu=mu,
                  sigma=sigma,
                  observed=log_y)

        trace = pm.sample(
            draws=250,
            tune=250,
            chains=2,
            target_accept=0.92,
            progressbar=False
        )

    future_log = np.log(FUTURE_DAYS)

    post = np.exp(
        trace.posterior["c"].values[...,None]
        + trace.posterior["alpha"].values[...,None]*future_log
    )

    return (
        post.mean((0,1)),
        np.percentile(post,10,(0,1)),
        np.percentile(post,90,(0,1))
    )


############################################
# RUN
############################################

if st.button("🚀 Run Forecast"):

    if iap_pos>=3:
        iap_mean,iap_low,iap_high = bayesian_power(x,y_iap,retention_alpha)
    else:
        iap_mean = iap_low = iap_high = np.zeros(len(FUTURE_DAYS))

    if ad_pos>=3:
        ad_mean,ad_low,ad_high = bayesian_power(x,y_ad,retention_alpha)
    else:
        ad_mean = ad_low = ad_high = np.zeros(len(FUTURE_DAYS))

    net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
    net_low = IAP_GROSS_TO_NET * iap_low + ad_low
    net_high = IAP_GROSS_TO_NET * iap_high + ad_high

    ############################################
    # TABLE
    ############################################

    df = pd.DataFrame({
        "Day":FUTURE_DAYS,
        "IAP":iap_mean,
        "AD":ad_mean,
        "NET":net_mean,
        "NET low":net_low,
        "NET high":net_high
    })

    st.subheader("Forecast")

    st.dataframe(
        df.round(2),
        hide_index=True,
        use_container_width=True
    )

    ############################################
    # GRAPH
    ############################################

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([FUTURE_DAYS,
                          FUTURE_DAYS[::-1]]),
        y=np.concatenate([net_high,
                          net_low[::-1]]),
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

    if ad_pos>=3:
        fig.add_trace(go.Scatter(
            x=FUTURE_DAYS,
            y=ad_mean,
            line=dict(dash="dot"),
            name="AD"
        ))

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=520
    )

    st.plotly_chart(fig,use_container_width=True)

