import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Bayesian power-law · IAP / AD separation · slope-based prior")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

IAP_MULTIPLIER = 3.0
AD_MULTIPLIER = 1.5

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

st.caption(f"ROAS_NET = {IAP_GROSS_TO_NET:.2f} × ROAS_IAP + ROAS_AD")

st.subheader("Input ROAS Values (Day 1–28)")

days_selected = st.multiselect(
    "ROAS girdiğin günler",
    options=list(range(1, 29)),
    default=[1, 3, 7, 14, 28]
)

if len(days_selected) < 3:
    st.warning("En az 3 gün seçmelisin.")
    st.stop()

roas_iap = {}
roas_ad = {}

for d in sorted(days_selected):
    c1, c2 = st.columns(2)
    with c1:
        roas_iap[d] = st.number_input(
            f"ROAS_IAP Day {d}",
            min_value=0.0,
            step=0.01,
            key=f"iap_{d}"
        )
    with c2:
        roas_ad[d] = st.number_input(
            f"ROAS_AD Day {d}",
            min_value=0.0,
            step=0.01,
            key=f"ad_{d}"
        )

x = np.array(sorted(days_selected))
y_iap = np.array([roas_iap[d] for d in x])
y_ad = np.array([roas_ad[d] for d in x])

if np.sum(y_iap > 0) < 3:
    st.error("IAP için en az 3 pozitif ROAS noktası gerekir.")
    st.stop()

if np.sum(y_ad > 0) < 3:
    st.error("AD için en az 3 pozitif ROAS noktası gerekir.")
    st.stop()

run_forecast = st.button("🚀 Run Bayesian Forecast")

@st.cache_resource
def bayesian_power_law(x, y, multiplier):
    x = np.array(x)
    y = np.array(y)

    mask = y > 0
    x = x[mask]
    y = y[mask]

    log_x = np.log(x)
    log_y = np.log(y)

    if 28 in x:
        roas_anchor = y[list(x).index(28)]
    else:
        roas_anchor = y[-1]

    target_180 = roas_anchor * multiplier
    prior_alpha = np.log(target_180 / roas_anchor) / np.log(180 / x.max())

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=prior_alpha, sigma=0.12)
        c = pm.Normal("c", mu=np.log(roas_anchor), sigma=0.35)
        sigma = pm.HalfNormal("sigma", 0.15)

        mu = c + alpha * log_x
        pm.Normal("obs", mu=mu, sigma=sigma, observed=log_y)

        trace = pm.sample(
            draws=400,
            tune=400,
            chains=2,
            target_accept=0.9,
            progressbar=False
        )

    future_log_x = np.log(FUTURE_DAYS)
    post = np.exp(
        trace.posterior["c"].values[..., None]
        + trace.posterior["alpha"].values[..., None] * future_log_x
    )

    mean = post.mean(axis=(0, 1))
    low = np.percentile(post, 10, axis=(0, 1))
    high = np.percentile(post, 90, axis=(0, 1))

    return mean, low, high

if not run_forecast:
    st.info("Bayesian forecast için Run Bayesian Forecast butonuna bas.")
    st.stop()

iap_mean, iap_low, iap_high = bayesian_power_law(
    tuple(x), tuple(y_iap), IAP_MULTIPLIER
)

ad_mean, ad_low, ad_high = bayesian_power_law(
    tuple(x), tuple(y_ad), AD_MULTIPLIER
)

net_mean = IAP_GROSS_TO_NET * iap_mean + ad_mean
net_low = IAP_GROSS_TO_NET * iap_low + ad_low
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

st.subheader("📈 ROAS Curves (Bayesian Power-law)")

fig, ax = plt.subplots()

ax.scatter(x, y_iap, color="blue", label="IAP Observed")
ax.scatter(x, y_ad, color="green", label="AD Observed")

ax.plot(FUTURE_DAYS, iap_mean, "--", color="blue", label="IAP Mean")
ax.plot(FUTURE_DAYS, ad_mean, "--", color="green", label="AD Mean")
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
