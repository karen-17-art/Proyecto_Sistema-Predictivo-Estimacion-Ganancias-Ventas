# =====================================================
# LIBRERÍAS
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="📊 Sistema Predictivo de Ganancia",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Sistema Predictivo de Ganancia por Venta")
st.write("Modelo con reglas de negocio para evitar resultados irreales.")

# =====================================================
# DATA
# =====================================================

url = "https://docs.google.com/spreadsheets/d/1RwKQhQbqvDNd3xJt5wiI6R4C7hAoc3TCBNWH3HcHO8g/export?format=csv"
data = pd.read_csv(url)

# =====================================================
# LIMPIEZA
# =====================================================

data["ganancia_total"] = (
    data["ganancia_total"]
    .astype(str)
    .str.replace(",", ".")
    .str.replace("%", "")
)

data["ganancia_total"] = pd.to_numeric(data["ganancia_total"], errors="coerce")

data["n_productos"] = pd.to_numeric(data["n_productos"], errors="coerce")
data["precio_promedio"] = pd.to_numeric(data["precio_promedio"], errors="coerce")
data["desc_promedio"] = pd.to_numeric(data["desc_promedio"], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

data = data[(data["desc_promedio"] >= 0) & (data["desc_promedio"] <= 1)]

# =====================================================
# FEATURE ENGINEERING
# =====================================================

data["log_productos"] = np.log1p(data["n_productos"])

data["ingreso_bruto"] = data["n_productos"] * data["precio_promedio"]

data["impacto_descuento"] = data["ingreso_bruto"] * data["desc_promedio"]

data["ingreso_neto"] = data["ingreso_bruto"] - data["impacto_descuento"]

data["en_campana"] = data["en_campana"].map({
    "True": 1,
    "False": 0,
    True: 1,
    False: 0
})

# =====================================================
# TARGET
# =====================================================

y = data["ganancia_total"]

# =====================================================
# FEATURES
# =====================================================

features = [
    "log_productos",
    "precio_promedio",
    "desc_promedio",
    "ingreso_bruto",
    "impacto_descuento",
    "ingreso_neto",
    "en_campana",
    "channel",
    "categoria_principal",
    "mes",
    "dia_semana"
]

X = data[features]
X = pd.get_dummies(X, drop_first=True)

# =====================================================
# TRAIN / TEST
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =====================================================
# MODELO
# =====================================================

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================================
# MÉTRICAS
# =====================================================

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

st.subheader("📊 Rendimiento del Modelo")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R²", f"{r2:.2f}")

# =====================================================
# INPUTS
# =====================================================

st.subheader("📝 Datos de la Venta")

col1, col2 = st.columns(2)

with col1:
    n_productos = st.number_input("📦 Número de Productos", 1, 1000, 5)
    precio_promedio = st.number_input("💰 Precio Promedio", 1.0, 10000.0, 100.0)
    desc_promedio = st.slider("🏷️ Descuento", 0.0, 1.0, 0.1)

with col2:
    channel = st.selectbox("📡 Canal", sorted(data["channel"].unique()))
    categoria_principal = st.selectbox("🛍️ Categoría", sorted(data["categoria_principal"].unique()))
    en_campana = st.selectbox("🔥 Campaña", [0, 1])

# =====================================================
# FEATURES INPUT
# =====================================================

log_productos = np.log1p(n_productos)

ingreso_bruto = n_productos * precio_promedio
impacto_descuento = ingreso_bruto * desc_promedio
ingreso_neto = ingreso_bruto - impacto_descuento

input_data = pd.DataFrame([{
    "log_productos": log_productos,
    "precio_promedio": precio_promedio,
    "desc_promedio": desc_promedio,
    "ingreso_bruto": ingreso_bruto,
    "impacto_descuento": impacto_descuento,
    "ingreso_neto": ingreso_neto,
    "en_campana": en_campana,
    "channel": channel,
    "categoria_principal": categoria_principal,
    "mes": data["mes"].mode()[0],
    "dia_semana": data["dia_semana"].mode()[0]
}])

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# =====================================================
# PREDICCIÓN
# =====================================================

if st.button("🔮 Predecir Ganancia"):

    pred = model.predict(input_data)[0]

    ingreso = ingreso_bruto
    descuento = impacto_descuento

    # =================================================
    # 🔥 REGLAS DE NEGOCIO (CLAVE FINAL)
    # =================================================

    ingreso_neto_real = ingreso - descuento

    ganancia = max(0, pred)

    # ❗ NO puede superar ingreso neto
    ganancia = min(ganancia, ingreso_neto_real * 0.95)

    # rentabilidad correcta
    rentabilidad = (ganancia / ingreso) * 100

    # evitar extremos
    rentabilidad = np.clip(rentabilidad, 1, 80)

    # =================================================
    # RESULTADOS
    # =================================================

    st.subheader("📊 Resultado de la Venta")

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Ingreso Bruto", f"S/ {ingreso:,.2f}")
    c2.metric("🏷️ Descuento", f"S/ {descuento:,.2f}")
    c3.metric("📈 Ganancia", f"S/ {ganancia:,.2f}")

    st.markdown("---")

    if rentabilidad >= 35:
        st.success("🟢 Alta rentabilidad")
    elif rentabilidad >= 15:
        st.warning("🟡 Rentabilidad media")
    else:
        st.error("🔴 Baja rentabilidad")

    # =================================================
    # GRÁFICO
    # =================================================

    fig, ax = plt.subplots()

    ax.bar(
        ["Ingreso Bruto", "Descuento", "Ganancia"],
        [ingreso, descuento, ganancia]
    )

    ax.set_title("📊 Análisis de Venta")

    st.pyplot(fig)

# =====================================================
# EJECUTAR
# =====================================================

# streamlit run app.py