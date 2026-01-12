import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Bayesian log-growth · IAP / AD separation · slope-based prior")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

IAP_MULTIPLIER = 5  
AD_MULTIPLIER  = 2.5

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
        "Custom IAP_GROSS_TO_NET (0–1)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01
    )

st.caption(
    f"ROAS_NET = {IAP_GROSS_TO_NET:.2f} × ROAS_IAP + ROAS_AD"
)

st.subheader("Input ROAS Values (Day 1–28)")

days_selected = st.multiselect(
    "ROAS girdiğin günler",
    options=list(range(1, 29)),
    default=[1, 3, 7, 14, 28]
)

if len(days_selected) < 3:
    st.warning("En az 3 gün seçmelisin.")
    st.stop()

roas_iap, roas_ad = {}, {}

for d in sorted(days_selected):
    c1, c2 = st.columns(2)
    with c1:
        roas_iap[d] = st.number_input(
            f"ROAS_IAP Day {d}",
            min_value=0.0, step=0.01, key=f"iap_{d}"
        )
    with c2:
        roas_ad[d] = st.number_input(
            f"ROAS_AD Day {d}",
            min_value=0.0, step=0.01, key=f"ad_{d}"
        )

x = np.array(sorted(days_selected))
y_iap = np.array([roas_iap[d] for d in x])
y_ad  = np.array([roas_ad[d]  for d in x])

def bayesian_log_growth(x, y, multiplier):
    log_x = np.log(x)

    roas_anchor = y[x == 28][0] if 28 in x else y[-1]
    target_180 = roas_anchor * multiplier

    prior_slope = (target_180 - roas_anchor) / (np.log(180) - np.log(28))

    with pm.Model() as model:
        a = pm.Normal("a", mu=prior_slope, sigma=abs(prior_slope) + 0.1)
        b = pm.Normal("b", mu=y[0], sigma=0.5)
        sigma = pm.HalfNormal("sigma", 0.1)

        mu = a * log_x + b
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            800,
            tune=800,
            chains=2,
            progressbar=False
        )

    future_log = np.log(FUTURE_DAYS)
    post = (
        trace.posterior["a"].values[..., None] * future_log
        + trace.posterior["b"].values[..., None]
    )

    mean = post.mean(axis=(0, 1))
    low  = np.percentile(post, 10, axis=(0, 1))
    high = np.percentile(post, 90, axis=(0, 1))

    return mean, low, high

iap_mean, iap_low, iap_high = bayesian_log_growth(
    x, y_iap, IAP_MULTIPLIER
)

ad_mean, ad_low, ad_high = bayesian_log_growth(
    x, y_ad, AD_MULTIPLIER
)

net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
net_low  = IAP_GROSS_TO_NET * iap_low  + ad_low
net_high = IAP_GROSS_TO_NET * iap_high + ad_high

df = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "ROAS_IAP": iap_mean.round(3),
    "ROAS_AD": ad_mean.round(3),
    "ROAS_NET": net_mean.round(3),
    "NET_low": net_low.round(3),
    "NET_high": net_high.round(3),
})

st.subheader("📊 Bayesian Long-Term ROAS Forecast")
st.dataframe(df, width="stretch")

st.subheader("📈 ROAS Curves (Bayesian)")

fig, ax = plt.subplots()

ax.scatter(x, y_iap, color="blue", label="IAP Observed")
ax.scatter(x, y_ad,  color="green", label="AD Observed")

ax.plot(FUTURE_DAYS, iap_mean, "--", color="blue", label="IAP Mean")
ax.plot(FUTURE_DAYS, ad_mean,  "--", color="green", label="AD Mean")
ax.plot(FUTURE_DAYS, net_mean, color="black", linewidth=2, label="NET Mean")

ax.fill_between(
    FUTURE_DAYS,
    net_low,
    net_high,
    color="gray",
    alpha=0.3,
    label="NET 10–90%"
)

ax.set_xlabel("Day")
ax.set_ylabel("ROAS")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.caption(
    f""" 
    IAP_GROSS_TO_NET: {IAP_GROSS_TO_NET:.2f}
    """
)

