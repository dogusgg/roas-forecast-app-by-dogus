import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

st.set_page_config(page_title="ROAS Long-Term Forecast", layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Retention-driven saturation model · No fake multipliers")

FUTURE_DAYS = np.array([90,120,180,360,720])

# -----------------------------
# RETENTION INPUT
# -----------------------------

st.subheader("Retention Inputs")

d1 = st.number_input("D1 Retention",0.0,1.0,0.40,0.01)
d7 = st.number_input("D7 Retention",0.0,1.0,0.20,0.01)
d28 = st.number_input("D28 Retention",0.0,1.0,0.10,0.01)

tail = (d7 + 2*d28)/3

# -----------------------------
# ROAS INPUT
# -----------------------------

st.subheader("ROAS Inputs")

days = np.array([1,3,7,14,28])

roas_iap = []

for d in days:
    val = st.number_input(f"ROAS_IAP Day {d}",0.0,5.0,0.0,0.01)
    roas_iap.append(val)

roas_iap = np.array(roas_iap)

mask = roas_iap > 0
x = days[mask]
y = roas_iap[mask]

if len(y) < 3:
    st.warning("At least 3 positive ROAS inputs required.")
    st.stop()

run = st.button("🚀 Generate Forecast")

if not run:
    st.stop()

# -----------------------------
# SATURATION MODEL
# -----------------------------

def sat_model(t, L, k, beta):
    return L * (1 - np.exp(-k * (t**beta)))

roas28 = y[-1]

# ceiling from retention
L_guess = roas28 * (1 + 3.5 * tail)

beta_guess = 0.6 + 0.8*d28
k_guess = 0.08

params,_ = curve_fit(
    sat_model,
    x,
    y,
    bounds=(
        [roas28*1.1, 0.001, 0.4],
        [roas28*6, 1, 2]
    ),
    p0=[L_guess,k_guess,beta_guess],
    maxfev=10000
)

L,k,beta = params

future = sat_model(FUTURE_DAYS,L,k,beta)

# -----------------------------
# CONFIDENCE
# -----------------------------

sigma = np.clip(0.25 - 0.15*tail,0.07,0.25)

low = future*(1-sigma)
high = future*(1+sigma)

df = pd.DataFrame({
    "Day":FUTURE_DAYS,
    "ROAS_IAP":future.round(3),
    "Low":low.round(3),
    "High":high.round(3)
})

st.subheader("Forecast")

st.dataframe(df,hide_index=True,use_container_width=True)

# -----------------------------
# GRAPH
# -----------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.concatenate([FUTURE_DAYS,FUTURE_DAYS[::-1]]),
    y=np.concatenate([high,low[::-1]]),
    fill='toself',
    fillcolor='rgba(150,150,150,0.25)',
    line=dict(color='rgba(255,255,255,0)'),
    name="Confidence"
))

fig.add_trace(go.Scatter(
    x=FUTURE_DAYS,
    y=future,
    mode="lines",
    line=dict(width=4),
    name="Forecast"
))

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=dict(size=9),
    name="Observed"
))

fig.update_layout(
    height=520,
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig,use_container_width=True)
