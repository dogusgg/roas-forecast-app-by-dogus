import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Sayfa Yapılandırması
st.set_page_config(page_title="ROAS Forecast Pro", page_icon="📈", layout="wide")

# Custom CSS - Daha modern bir görünüm için
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stNumberInput { border-radius: 8px; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 ROAS Long-Term Forecasting Tool")
st.caption("Early signals'tan (D1-D28) 720 güne kadar akıllı projeksiyon.")

# -------------------------
# 1. INPUT ALANI (Ana Ekran)
# -------------------------
with st.container():
    st.subheader("📊 Short-Term ROAS Inputs")
    st.info("💡 En az 3 adet değer girin. Boş bırakılan günler tahminde dikkate alınmaz.")
    
    # Veri girişini 7'li sütunlara bölerek daha temiz bir tablo görünümü sağlıyoruz
    input_data = {}
    for row in range(4): # 4 satır x 7 sütun = 28 gün
        cols = st.columns(7)
        for col_idx in range(7):
            day = row * 7 + col_idx + 1
            with cols[col_idx]:
                val = st.number_input(f"Day {day}", min_value=0.0, step=0.01, key=f"d{day}", format="%.2f")
                if val > 0:
                    input_data[day] = val

# -------------------------
# 2. AYARLAR & HESAPLAMA
# -------------------------
st.divider()

if len(input_data) < 3:
    st.warning("👉 Devam etmek için en az 3 adet veri girişi yapmalısınız.")
    st.stop()

days = np.array(sorted(input_data.keys()))
roas_values = np.array([input_data[d] for d in days])
n_points = len(days)

# Model Seçimi
if n_points <= 4:
    model_type = "Log-Linear (Stabil)"
    X, y = np.log(days).reshape(-1, 1), roas_values
elif n_points <= 9:
    model_type = "Power Law (Agresif)"
    X, y = np.log(days).reshape(-1, 1), np.log(roas_values)
else:
    model_type = "Saturation (Doygunluk)"
    X, y = (1 / days).reshape(-1, 1), roas_values

# Fit & Predict
model = LinearRegression().fit(X, y)
future_days = np.array([90, 120, 180, 360, 720])
future_X = np.log(future_days).reshape(-1, 1) if "Saturation" not in model_type else (1 / future_days).reshape(-1, 1)

if "Power" in model_type:
    base_pred = np.exp(model.predict(future_X))
    fitted = np.exp(model.predict(X))
else:
    base_pred = model.predict(future_X)
    fitted = model.predict(X)

# Belirsizlik Hesapları
error_factor = min(0.3, 1.2 / np.sqrt(n_points))
best_case = base_pred * (1 + error_factor)
worst_case = base_pred * (1 - error_factor)

# -------------------------
# 3. GÖRSELLEŞTİRME (PLOTLY)
# -------------------------
fig = go.Figure()

# Gerçek Veri
fig.add_trace(go.Scatter(x=days, y=roas_values, name='Actual ROAS', mode='lines+markers', line=dict(color='#FF4B4B', width=4)))

# Senaryolar
fig.add_trace(go.Scatter(x=future_days, y=base_pred, name='Base Case', line=dict(color='#1F77B4', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=future_days, y=best_case, name='Best Case', line=dict(color='#2CA02C', width=1, dash='dot')))
fig.add_trace(go.Scatter(x=future_days, y=worst_case, name='Worst Case', line=dict(color='#FF7F0E', width=1, dash='dot')))

# Confidence Alanı
fig.add_trace(go.Scatter(
    x=np.concatenate([future_days, future_days[::-1]]),
    y=np.concatenate([best_case, worst_case[::-1]]),
    fill='toself', fillcolor='rgba(31, 119, 180, 0.1)', line=dict(color='rgba(255,255,255,0)'),
    name='Risk Zone', hoverinfo="skip"
))

fig.update_layout(
    template="simple_white",
    hovermode="x unified",
    xaxis=dict(type="log", title="Days (Log Scale)", gridcolor="#f0f0f0"),
    yaxis=dict(title="ROAS Value", gridcolor="#f0f0f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=50, b=0),
    height=500
)

# Sol tarafta metrikler, sağ tarafta grafik
c1, c2 = st.columns([1, 3])
with c1:
    st.metric("Model Rejimi", model_type.split(" ")[0])
    st.metric("Veri Gücü", f"{n_points} Nokta")
    st.metric("Tahmin Belirsizliği", f"±%{error_factor*100:.0f}")
    
    res_df = pd.DataFrame({"Day": future_days, "ROAS": base_pred.round(2)})
    st.dataframe(res_df, hide_index=True, use_container_width=True)

with c2:
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 4. EXPORT
# -------------------------
st.divider()
st.download_button("📥 Tahminleri CSV Olarak İndir", 
                   pd.DataFrame({"Day": future_days, "Worst": worst_case, "Base": base_pred, "Best": best_case}).to_csv(index=False), 
                   "roas_forecast.csv")
