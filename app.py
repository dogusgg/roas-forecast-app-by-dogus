import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Bayesian power-law · IAP / AD separation")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

IAP_MULTIPLIER = 3.0
AD_MULTIPLIER = 1.5

st.subheader("Revenue Parameters")

fee_option = st.selectbox("IAP_GROSS_TO_NET", ["70%", "85%", "Custom"])

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
        roas_iap[d] = st.number_input(f"ROAS_IAP Day {d}", min_value=0.0, step=0.01)
    with c2:
        roas_ad[d] = st.number_input(f"ROAS_AD Day {d}", min_value=0.0, step=0.01)

x = np.array(sorted(days_selected))
y_iap = np.array([roas_iap[d] for d in x])
y_ad = np.array([roas_ad[d] for d in x])

iap_pos = np.sum(y_iap > 0)
ad_pos = np.sum(y_ad > 0)

st.caption("ℹ️ IAP forecast için **en az 3 pozitif ROAS** gereklidir.")
st.caption("ℹ️ AD forecast için **en az 3 pozitif ROAS** gereklidir.")

if iap_pos < 3:
    st.warning("IAP için yeterli pozitif veri yok. IAP forecast yapılmayacak.")
if ad_pos < 3:
    st.warning("AD için yeterli pozitif veri yok. AD forecast yapılmayacak.")

run_enabled = (iap_pos >= 3) or (ad_pos >= 3)

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
        anchor = y[list(x).index(28)]
    else:
        anchor = y[-1]

    target_180 = anchor * multiplier
    alpha_prior = np.log(target_180 / anchor) / np.log(180 / x.max())

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=alpha_prior, sigma=0.12)
        c = pm.Normal("c", mu=np.log(anchor), sigma=0.35)
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

    future_log = np.log(FUTURE_DAYS)
    post = np.exp(
        trace.posterior["c"].values[..., None]
        + trace.posterior["alpha"].values[..., None] * future_log
    )

    return (
        post.mean(axis=(0, 1)),
        np.percentile(post, 10, axis=(0, 1)),
        np.percentile(post, 90, axis=(0, 1))
    )

if not run_enabled:
    st.info("Forecast çalıştırmak için IAP veya AD’den en az biri ≥3 pozitif ROAS içermelidir.")
    st.stop()

run_forecast = st.button("🚀 Run Bayesian Forecast")

if not run_forecast:
    st.stop()

if iap_pos >= 3:
    iap_mean, iap_low, iap_high = bayesian_power_law(tuple(x), tuple(y_iap), IAP_MULTIPLIER)
else:
    iap_mean = iap_low = iap_high = np.zeros(len(FUTURE_DAYS))

if ad_pos >= 3:
    ad_mean, ad_low, ad_high = bayesian_power_law(tuple(x), tuple(y_ad), AD_MULTIPLIER)
else:
    ad_mean = ad_low = ad_high = np.zeros(len(FUTURE_DAYS))

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

st.subheader("📈 ROAS Curves")

fig, ax = plt.subplots()

ax.scatter(x, y_iap, color="blue", label="IAP Observed")
ax.scatter(x, y_ad, color="green", label="AD Observed")

ax.plot(FUTURE_DAYS, iap_mean, "--", color="blue", label="IAP Forecast")
ax.plot(FUTURE_DAYS, ad_mean, "--", color="green", label="AD Forecast")
ax.plot(FUTURE_DAYS, net_mean, color="black", linewidth=2, label="NET Forecast")

ax.fill_between(FUTURE_DAYS, net_low, net_high, color="gray", alpha=0.3)

ax.legend()
ax.grid(True)

st.pyplot(fig)
