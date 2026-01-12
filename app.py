import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("ROAS_IAP / ROAS_AD separation → ROAS_NET projection (log-growth based)")

NET_FEE = 0.7
FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

st.subheader("Forecast Mode")

mode = st.radio(
    "Projection assumption",
    ["Conservative", "Base", "Aggressive"],
    horizontal=True,
)

MODE_MULTIPLIER = {
    "Conservative": 1.6,   
    "Base":         2.0,  
    "Aggressive":   2.4,  
}

st.caption(
    """
    **Conservative:** Data + mild long-term tail  
    **Base:** Industry-average monetization tail  
    **Aggressive:** Strong LiveOps / hit-level monetization  
    """
)

st.subheader("Input ROAS Values (Day 1–28)")
st.caption("En az **3 gün** için ROAS_IAP ve ROAS_AD girilmelidir. 0 girilebilir.")

days_selected = st.multiselect(
    "ROAS girdiğin günler",
    options=list(range(1, 29)),
    default=[1, 3, 7]
)

if len(days_selected) < 3:
    st.warning("En az 3 gün seçmelisin.")
    st.stop()

roas_iap = {}
roas_ad = {}

for d in sorted(days_selected):
    col1, col2 = st.columns(2)

    with col1:
        roas_iap[d] = st.number_input(
            f"ROAS_IAP Day {d}",
            min_value=0.0,
            step=0.01,
            key=f"iap_{d}"
        )

    with col2:
        roas_ad[d] = st.number_input(
            f"ROAS_AD Day {d}",
            min_value=0.0,
            step=0.01,
            key=f"ad_{d}"
        )

x = np.array(sorted(days_selected))
y_iap = np.array([roas_iap[d] for d in x])
y_ad  = np.array([roas_ad[d]  for d in x])

if np.any(y_iap < 0) or np.any(y_ad < 0):
    st.error("ROAS değerleri negatif olamaz.")
    st.stop()

n_points = len(x)

def log_model_fit(x, y):
    coef = np.polyfit(np.log(x), y, 1)
    return lambda d: coef[0] * np.log(d) + coef[1]

iap_predict = log_model_fit(x, y_iap)
ad_predict  = log_model_fit(x, y_ad)

iap_pred_raw = iap_predict(FUTURE_DAYS)
ad_pred_raw  = ad_predict(FUTURE_DAYS)

# Negatifleri engelle (başka cap YOK)
iap_pred_raw = np.maximum(iap_pred_raw, 0)
ad_pred_raw  = np.maximum(ad_pred_raw, 0)

iap_pred = iap_pred_raw * MODE_MULTIPLIER[mode]
ad_pred  = ad_pred_raw  * MODE_MULTIPLIER[mode]

net_pred = NET_FEE * iap_pred + ad_pred

base_error = 0.12
data_penalty = max(0, (5 - n_points)) * 0.06
horizon_penalty = np.log(FUTURE_DAYS / max(x)) * 0.05

total_error = base_error + data_penalty + horizon_penalty

net_low  = net_pred * (1 - total_error)
net_high = net_pred * (1 + total_error)

df = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "ROAS_IAP": iap_pred.round(3),
    "ROAS_AD": ad_pred.round(3),
    "ROAS_NET": net_pred.round(3),
    "NET_low": net_low.round(3),
    "NET_high": net_high.round(3),
})

st.subheader("📊 Long-Term ROAS Forecast")
st.dataframe(df, width="stretch")

if n_points <= 4:
    st.error("⚠️ Low confidence – very limited data")
elif n_points <= 6:
    st.warning("⚠️ Medium confidence")
else:
    st.success("✅ High confidence")

st.caption(
    f"""
    **Forecast mode:** {mode}  
    **Model:** Log-growth (IAP + AD)  
    **ROAS_NET = {NET_FEE} × ROAS_IAP + ROAS_AD**
    """
)

st.subheader("📈 ROAS Curves")

fig, ax = plt.subplots()

ax.scatter(x, y_iap, color="blue", label="IAP Observed")
ax.scatter(x, y_ad,  color="green", label="AD Observed")

ax.plot(FUTURE_DAYS, iap_pred, color="blue", linestyle="--", label="IAP Forecast")
ax.plot(FUTURE_DAYS, ad_pred,  color="green", linestyle="--", label="AD Forecast")
ax.plot(FUTURE_DAYS, net_pred, color="black", linewidth=2, label="NET Forecast")

ax.fill_between(
    FUTURE_DAYS,
    net_low,
    net_high,
    color="gray",
    alpha=0.3,
    label="NET Confidence Band"
)

ax.set_xlabel("Day")
ax.set_ylabel("ROAS")
ax.legend()
ax.grid(True)

st.pyplot(fig)
