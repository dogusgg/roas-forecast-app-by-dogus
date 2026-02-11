import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sayfa Ayarları
st.set_page_config(page_title="ROAS Forecasting Tool", layout="wide")

st.title("📈 ROAS Long-Term Forecasting Tool")
st.markdown("""
Early ROAS verilerini girerek 720 güne kadar projeksiyon oluşturun. 
Model, veri miktarına göre en uygun algoritmayı (Log, Power veya Saturation) otomatik seçer.
""")

# -------------------------------------------------
# 1. INPUT BÖLÜMÜ (Day 1–28 Esnek Giriş)
# -------------------------------------------------
st.sidebar.header("📊 Veri Girişi")
st.sidebar.info("En az 3 adet ROAS değeri girilmelidir.")

roas_dict = {}
# 1'den 28'e kadar olan günler için input alanları
for day in range(1, 29):
    # Sadece yaygın kullanılan günleri default açık gösterelim, diğerlerini gizleyelim (opsiyonel tasarım)
    default_val = 0.0
    val = st.sidebar.number_input(f"Day {day} ROAS", min_value=0.0, value=default_val, step=0.01, key=f"d{day}")
    if val > 0:
        roas_dict[day] = val

# Validasyon
if len(roas_dict) < 3:
    st.warning("⚠️ Lütfen tahmin üretmek için en az 3 adet ROAS değeri girin.")
    st.stop()

days = np.array(sorted(roas_dict.keys()))
roas_values = np.array([roas_dict[d] for d in days])
n_points = len(days)

# -------------------------------------------------
# 2. OTOMATİK MODEL SEÇİMİ VE FIT
# -------------------------------------------------
# Veri miktarına göre rejim belirleme
if n_points <= 4:
    model_type = "Log-Linear"
elif n_points <= 9:
    model_type = "Power Law"
else:
    model_type = "Saturation"

future_days = np.array([90, 120, 180, 360, 720])

# Model Hazırlığı
if model_type == "Log-Linear":
    X = np.log(days).reshape(-1, 1)
    future_X = np.log(future_days).reshape(-1, 1)
    y = roas_values
elif model_type == "Power Law":
    X = np.log(days).reshape(-1, 1)
    future_X = np.log(future_days).reshape(-1, 1)
    y = np.log(roas_values) # Log-Log fit
else: # Saturation (1/x)
    X = (1 / days).reshape(-1, 1)
    future_X = (1 / future_days).reshape(-1, 1)
    y = roas_values

model = LinearRegression()
model.fit(X, y)

# Tahminleri Hesapla
if model_type == "Power Law":
    base_pred = np.exp(model.predict(future_X))
    fitted = np.exp(model.predict(X.reshape(-1, 1)))
else:
    base_pred = model.predict(future_X)
    fitted = model.predict(X.reshape(-1, 1))

# -------------------------------------------------
# 3. BAYESIAN CONFIDENCE & ERROR SIMULATION
# -------------------------------------------------
# Residual (artık) analizi ile belirsizlik hesabı
residuals = roas_values - fitted
residual_std = np.std(residuals) if len(residuals) > 1 else 0.1
posterior_std = residual_std * np.sqrt(1 + 1 / n_points)

# %95 Güven Aralığı (Bayesian Approx)
lower_conf = base_pred - (1.96 * posterior_std)
upper_conf = base_pred + (1.96 * posterior_std)

# Error Simulation (Best/Worst Case - Manuel Çarpan)
error_factor = min(0.3, 1 / np.sqrt(n_points)) # Veri arttıkça daralan hata payı
best_case = base_pred * (1 + error_factor)
worst_case = base_pred * (1 - error_factor)

# -------------------------------------------------
# 4. GÖRSELLEŞTİRME VE SONUÇLAR
# -------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Tahmin Tablosu")
    result_df = pd.DataFrame({
        "Day": future_days,
        "Worst Case": worst_case.round(3),
        "Base Case": base_pred.round(3),
        "Best Case": best_case.round(3),
        "Lower (Bayes)": np.maximum(lower_conf, 0).round(3),
        "Upper (Bayes)": upper_conf.round(3)
    })
    st.dataframe(result_df, use_container_width=True)
    
    st.info(f"""
    **Seçilen Model:** {model_type}  
    **Veri Noktası:** {n_points} gün  
    **Hata Payı:** ±%{error_factor*100:.1f}
    """)

with col2:
    st.subheader("📈 Projeksiyon Grafiği")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Mevcut Veri
    ax.plot(days, roas_values, 'ro-', label="Gerçekleşen ROAS", linewidth=2)
    
    # Tahminler
    ax.plot(future_days, base_pred, 'b--', label="Base Projeksiyon", alpha=0.8)
    ax.plot(future_days, best_case, 'g:', label="Best Case", alpha=0.6)
    ax.plot(future_days, worst_case, 'r:', label="Worst Case", alpha=0.6)
    
    # Güven Aralığı (Gölge)
    ax.fill_between(future_days, np.maximum(lower_conf, 0), upper_conf, color='gray', alpha=0.2, label="Bayesian Confidence")
    
    ax.set_xscale('log') # Zaman logaritmik eksende daha iyi okunur
    ax.set_xlabel("Gün (Log Scale)")
    ax.set_ylabel("ROAS")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    st.pyplot(fig)

# CSV İndirme Butonu
csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("📊 Verileri CSV Olarak İndir", csv, "roas_forecast.csv", "text/csv")
