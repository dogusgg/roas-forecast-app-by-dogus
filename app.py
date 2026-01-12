import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------- Page setup ----------------
st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("ROAS_IAP / ROAS_AD separation → ROAS_NET projection with Forecast Mode")

NET_FEE = 0.7
FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

# ---------------- Forecast mode ----------------
st.subheader("Forecast Mode")

mode = st.radio(
    "Projection assumption",
    ["Conservative", "Base", "Aggressive"],
    horizontal=True,
)

MODE_MULTIPLIER = {
    "Conservative": 1.8,
    "Base": 2.2,
    "Aggressive": 2.6,
}

st.caption(
    f"""
    **Conservative:** Data-only (no uplift)  
    **Base:** Typical industry tail  
    **Aggressive:** Strong LiveOps / monetization tail  
    """
)

# ---------------- Input ----------------
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

# ---------------- Validation ----------------
x = np.array(sorted(days_selected))
y_iap = np.array([roas_iap[d] for d in x])
y_ad  = np.array([roas_ad[d]  for d in x])

if np.any(y_iap < 0) or np.any(y_ad < 0):
    st.error("ROAS değerleri negatif olamaz.")
    st.stop()

n_points = len(x)

# ---------------- Model helpers ----------------
def log_model_fit(x, y):
    coef = np.polyfit(np.log(x), y, 1)
    return lambda d: coef[0] * np.log(d) + coef[1]

def anchored_saturation_fit(x, y):
    base = max(y)

    def sat_fn(d, a, b):
        return base + a * (1 - np.exp(-b * d))

    popt, _ = curve_fit(
        sat_fn,
        x,
        y,
        bounds=([0, 0], [base * 6, 1]),
        maxfev=10000
    )
    return lambda d: sat_fn(d, *popt)

# ---------------- Model selection ----------------
def choose_model(x, y, component_name):
    if len(x) <= 5:
        return "Log Growth", log_model_fit(x, y)
    else:
        if component_name == "AD":
            return "Log Growth", log_model_fit(x, y)
        else:
            return "Anchored Saturation", anchored_saturation_fit(x, y)

# ---------------- Fit models ----------------
iap_model_name, iap_predict = choose_model(x, y_iap, "IAP")
ad_model_name,  ad_predict  = choose_model(x, y_ad,  "AD")

iap_pred_raw = iap_predict(FUTURE_DAYS)
ad_pred_raw  = ad_predict(FUTURE_DAYS)

# Safety floor
iap_pred_raw = np.maximum(iap_pred_raw, y_iap[-1])
ad_pred_raw  = np.maximum(ad_pred_raw, y_ad[-1])

# ---------------- NET ROAS (base) ----------------
net_pred_base = NET_FEE * iap_pred_raw + ad_pred_raw

# ---------------- Apply forecast mode ----------------
net_pred = net_pred_base * MODE_MULTIPLIER[mode]

# ---------------- Confidence band ----------------
base_error = 0.12
data_penalty = max(0, (5 - n_points)) * 0.06
horizon_penalty = np.log(FUTURE_DAYS / max(x)) * 0.05

total_error = base_error + data_penalty + horizon_penalty

net_low  = net_pred * (1 - total_error)
net_high = net_pred * (1 + total_error)

# ---------------- Output table ----------------
df = pd.DataFrame({
    "Day": FUTURE_DAYS,
    "ROAS_IAP": iap_pred_raw.round(3),
    "ROAS_AD": ad_pred_raw.round(3),
    "ROAS_NET": net_pred.round(3),
    "NET_low": net_low.round(3),
    "NET_high": net_high.round(3),
})

st.subheader("📊 Long-Term ROAS Forecast")
st.dataframe(df, width="stretch")

# ---------------- Confidence badge ----------------
if n_points <= 4:
    st.error("⚠️ Low confidence – very limited data")
elif n_points <= 6:
    st.warning("⚠️ Medium confidence")
else:
    st.success("✅ High confidence")

st.caption(
    f"""
    **Forecast mode:** {mode}  
    **IAP model:** {iap_model_name}  
    **AD model:** {ad_model_name}  
    **ROAS_NET = {NET_FEE} × ROAS_IAP + ROAS_AD × mode multiplier**
    """
)

# ---------------- Plot ----------------
st.subheader("📈 ROAS Curves")

fig, ax = plt.subplots()

ax.scatter(x, y_iap, color="blue", label="IAP Observed")
ax.scatter(x, y_ad,  color="green", label="AD Observed")

ax.plot(FUTURE_DAYS, iap_pred_raw, color="blue", linestyle="--", label="IAP Forecast")
ax.plot(FUTURE_DAYS, ad_pred_raw,  color="green", linestyle="--", label="AD Forecast")
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

