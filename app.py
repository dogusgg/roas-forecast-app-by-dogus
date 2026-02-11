import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Sayfa Ayarları
st.set_page_config(page_title="ROAS Forecasting Tool", layout="wide")

st.title("📈 ROAS Long-Term Forecasting Tool")
st.markdown("Interaktif grafik ve detaylı senaryo analizleri ile uzun vadeli ROAS projeksiyonu.")

# -------------------------------------------------
# 1. SIDEBAR - VERİ GİRİŞİ
# -------------------------------------------------
st.sidebar.header("📊 Veri Girişi")
st.sidebar.info("En az 3 adet ROAS değeri girilmelidir.")

roas_dict = {}
cols = st.sidebar.columns(2)
for day in range(1, 29):
    # Kullanıcıyı yormamak için sidebar'da düzenli gösterelim
    with cols[day % 2]:
        val = st.number_input(f"D{day}", min_value=0.0, value=0.0, step=0.01, key=f"d{day}")
        if val > 0:
            roas_dict[day] = val

if len(roas_dict) < 3:
    st.warning("⚠️ Lütfen tahmin üretmek için en az 3 adet ROAS değeri girin.")
    st.stop()

days = np.array(sorted(roas_dict.keys()))
roas_values = np.array([roas_dict[d] for d in days])
n_points = len(days)

# -------------------------------------------------
# 2. MODEL SEÇİMİ VE HESAPLAMA
# -------------------------------------------------
# Otomatik Model Seçimi (Regime Detection)
if n_points <= 4:
    model_type = "Log-Linear"
elif n_points <= 9:
    model_type = "Power Law"
else:
    model_type = "Saturation"

future_days = np.array([90, 120, 180, 360, 720])

if model_type == "Log-Linear":
    X = np.log(days).reshape(-1, 1)
    future_X = np.log(future_days).reshape(-1, 1)
    y = roas_values
elif model_type == "Power Law":
    X = np.log(days).reshape(-1, 1)
    future_X = np.log(future_days).reshape(-1, 1)
    y = np.log(roas_values)
else: # Saturation
    X = (1 / days).reshape(-1, 1)
    future_X = (1 / future_days).reshape(-1, 1)
    y = roas_values

model = LinearRegression()
model.fit(X, y)

if model_type == "Power Law":
    base_pred = np.exp(model.predict(future_X))
    fitted = np.exp(model.predict(X))
else:
    base_pred = model.predict(future_X)
    fitted = model.predict(X)

# Bayesian & Error Simulation
residuals = roas_values - fitted
residual_std = np.std(residuals) if n_points > 1 else 0.1
posterior_std = residual_std * np.sqrt(1 + 1 / n_points)

error_factor = min(0.3, 1 / np.sqrt(n_points))
best_case = base_pred * (1 + error_factor)
worst_case = base_pred * (1 - error_factor)
lower_conf = np.maximum(base_pred - (1.96 * posterior_std), 0)
upper_conf = base_pred + (1.96 * posterior_std)

# -------------------------------------------------
# 3. INTERAKTIF GRAFİK (PLOTLY)
# -------------------------------------------------
fig = go.Figure()

# Gerçek Veri (Noktalar)
fig.add_trace(go.Scatter(
    x=days, y=roas_values,
    mode='markers+lines', name='Gerçekleşen ROAS',
    line=dict(color='red', width=3),
    marker=dict(size=10)
))

# Base Forecast
fig.add_trace(go.Scatter(
    x=future_days, y=base_pred,
    mode='lines+markers', name='Base Case',
    line=dict(color='blue', dash='dash')
))

# Best Case
fig.add_trace(go.Scatter(
    x=future_days, y=best_case,
    mode='lines', name='Best Case',
    line=dict(color='green', width=1, dash='dot')
))

# Worst Case
fig.add_trace(go.Scatter(
    x=future_days, y=worst_case,
    mode='lines', name='Worst Case',
    line=dict(color='orange', width=1, dash='dot')
))

# Bayesian Confidence Band (Gölge Alan)
fig.add_trace(go.Scatter(
    x=np.concatenate([future_days, future_days[::-1]]),
    y=np.concatenate([upper_conf, lower_conf[::-1]]),
    fill='toself', fillcolor='rgba(128,128,128,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", showlegend=True, name='Bayesian Confidence'
))

fig.update_layout(
    title=f"ROAS Projeksiyonu ({model_type} Modeli)",
    xaxis_title="Gün (Log Scale)", yaxis_title="ROAS",
    xaxis_type="log", hovermode="x unified",
    template="plotly_white", height=600
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 4. TABLO VE EXPORT
# -------------------------------------------------
res_df = pd.DataFrame({
    "Gün": future_days,
    "Worst Case": worst_case.round(3),
    "Base Case": base_pred.round(3),
    "Best Case": best_case.round(3)
})

st.subheader("📋 Senaryo Detayları")
st.table(res_df)

st.download_button("📥 Sonuçları CSV Olarak İndir", res_df.to_csv(index=False).encode('utf-8'), "roas_forecast.csv")
