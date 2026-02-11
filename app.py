import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Forecast",layout="centered")

st.title("📈 ROAS Long-Term Forecast")
st.caption("Retention-weighted Hill model")

FUTURE = np.array([90,120,180,360,720])

# ------------------------
# RETENTION
# ------------------------

st.subheader("Retention")

d1 = st.number_input("D1",0.0,1.0,0.40,0.01)
d7 = st.number_input("D7",0.0,1.0,0.20,0.01)
d28 = st.number_input("D28",0.0,1.0,0.10,0.01)

tail = (d7 + 3*d28)/4

# ------------------------
# ROAS INPUT
# ------------------------

days = np.array([1,3,7,14,28])

st.subheader("ROAS IAP")

iap=[]
for d in days:
    iap.append(st.number_input(f"IAP Day {d}",0.0,5.0,0.0,0.01))

iap=np.array(iap)

st.subheader("ROAS AD")

ad=[]
for d in days:
    ad.append(st.number_input(f"AD Day {d}",0.0,5.0,0.0,0.01))

ad=np.array(ad)

run=st.button("🚀 Generate Forecast")

if not run:
    st.stop()

# ------------------------
# HILL MODEL
# ------------------------

def hill(t,L,beta,h):
    return L*(t**beta)/(t**beta + h**beta)

def forecast(roas,curve_type="iap"):

    mask=roas>0
    x=days[mask]
    y=roas[mask]

    if len(y)<3:
        return np.zeros_like(FUTURE)

    roas7 = y[min(2,len(y)-1)]
    roas28 = y[-1]

    momentum = roas28 / max(roas7,0.01)

    lifetime_mult = 1.8 + 4.5*tail + 0.6*np.log(max(momentum,1.01))

    L = roas28 * lifetime_mult

    beta = (1.3 if curve_type=="iap" else 1.0) + 1.5*d28

    h = 28

    return hill(FUTURE,L,beta,h)

iap_f = forecast(iap,"iap")
ad_f = forecast(ad,"ad")

net = 0.7*iap_f + ad_f

sigma = np.clip(0.22 - tail*0.12,0.08,0.22)

low = net*(1-sigma)
high = net*(1+sigma)

df=pd.DataFrame({
    "Day":FUTURE,
    "IAP":iap_f.round(3),
    "AD":ad_f.round(3),
    "NET":net.round(3),
    "LOW":low.round(3),
    "HIGH":high.round(3)
})

st.subheader("Forecast")

st.dataframe(df,hide_index=True,use_container_width=True)

# ------------------------
# GRAPH
# ------------------------

fig=go.Figure()

fig.add_trace(go.Scatter(
    x=np.concatenate([FUTURE,FUTURE[::-1]]),
    y=np.concatenate([high,low[::-1]]),
    fill='toself',
    fillcolor='rgba(150,150,150,0.25)',
    line=dict(color='rgba(255,255,255,0)')
))

fig.add_trace(go.Scatter(x=FUTURE,y=net,mode="lines",name="NET",line=dict(width=4)))
fig.add_trace(go.Scatter(x=FUTURE,y=iap_f,mode="lines",name="IAP",line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=FUTURE,y=ad_f,mode="lines",name="AD",line=dict(dash="dot")))

# observed only positive
mask_iap=iap>0
fig.add_trace(go.Scatter(x=days[mask_iap],y=iap[mask_iap],mode="markers",name="Observed IAP"))

fig.update_layout(template="plotly_white",height=520)

st.plotly_chart(fig,use_container_width=True)
