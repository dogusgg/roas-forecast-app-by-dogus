import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="ROAS Prediction Engine", layout="wide")

st.title("📈 ROAS Prediction Engine")
st.caption("Final Edition · Net-Targeted · IAP & AD Integrated")

FUTURE_DAYS = np.array([90, 120, 180, 360, 720])

# ==========================================
# 1. INPUT SECTION (TAM UI)
# ==========================================
col_top1, col_top2 = st.columns([1, 2])

with col_top1:
    st.subheader("1. Profitability")
    fee_mode = st.selectbox("Store Fee Mode", ["Standard (30%)", "SMB (15%)", "Custom"])
    if fee_mode == "Standard (30%)": GROSS_TO_NET = 0.70
    elif fee_mode == "SMB (15%)": GROSS_TO_NET = 0.85
    else: GROSS_TO_NET = st.number_input("Custom Factor", 0.0, 1.0, 0.70)

with col_top2:
    st.subheader("2. Retention Metrics")
    sel_ret_days = st.multiselect("Select Retention Days", [1, 7, 28], default=[1, 7, 28])
    ret_data = {}
    r_cols = st.columns(len(sel_ret_days))
    for i, d in enumerate(sorted(sel_ret_days)):
        with r_cols[i]:
            def_v = {1:0.40, 7:0.20, 28:0.10}.get(d, 0.0)
            ret_data[d] = st.number_input(f"D{d} Ret", 0.0, 1.0, def_v, 0.01)

st.subheader("3. ROAS Data Points")
sel_roas_days = st.multiselect("Select ROAS Days", [1, 3, 7, 14, 28], default=[1, 3, 7, 14, 28])

roas_iap, roas_ad = {}, {}
for d in sorted(sel_roas_days):
    c1, c2 = st.columns(2)
    with c1:
        def_iap = {1:0.02, 3:0.05, 7:0.10, 14:0.16, 28:0.25}.get(d, 0.0)
        roas_iap[d] = st.number_input(f"Day {d} IAP ROAS (Gross)", 0.0, 10.0, def_iap, 0.01)
    with c2:
        roas_ad[d] = st.number_input(f"Day {d} AD ROAS", 0.0, 10.0, 0.0, 0.01)

x_days = np.array(sorted(sel_roas_days))
y_iap = np.array([roas_iap[d] for d in x_days])
y_ad = np.array([roas_ad[d] for d in x_days])

run = st.button("🚀 GENERATE FORECAST", use_container_width=True, type="primary")

if not run: st.stop()

# ==========================================
# 2. CALIBRATED ENGINE
# ==========================================

def calculate_retention_score(ret_dict):
    return (0.6
