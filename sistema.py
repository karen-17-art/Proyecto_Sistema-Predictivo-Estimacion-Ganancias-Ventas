import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------
# CARGAR DATOS
# -------------------------
data = pd.read_excel("data_final.xlsx")
data = data.dropna()

# -------------------------
# VARIABLES
# -------------------------
X = data[['unit_price', 'porcentaje_descuento', 'quantity', 'margen_unitario']]
y = data['ganancia']

# -------------------------
# MODELO
# -------------------------
modelo = LinearRegression()
modelo.fit(X, y)

# -------------------------
# INTERFAZ
# -------------------------
st.title("📊 Sistema Predictivo de Ganancia")

st.write("Ingrese los datos del producto:")

precio = st.number_input("Precio de venta (unit_price)", min_value=0.0)
descuento = st.number_input("Porcentaje de descuento (%)", min_value=0.0, max_value=100.0)
cantidad = st.number_input("Cantidad", min_value=1)
margen = st.number_input("Margen unitario", min_value=0.0)

# -------------------------
# BOTÓN (TODO VA DENTRO)
# -------------------------
if st.button("Predecir ganancia"):

    entrada = pd.DataFrame({
        'unit_price': [precio],
        'porcentaje_descuento': [descuento],
        'quantity': [cantidad],
        'margen_unitario': [margen]
    })

    prediccion = modelo.predict(entrada)[0]

    # Resultado
    st.write(f"💰 Ganancia estimada: S/ {prediccion:.2f}")

    # -------------------------
    # LÓGICA GANAR / PERDER
    # -------------------------
    if prediccion > 0:
        st.success("✅ Genera ganancia")
        st.info("💡 Recomendación: Este descuento es viable")

    elif prediccion < 0:
        st.error("❌ Genera pérdida")
        st.warning("⚠️ Recomendación: No aplicar este descuento")

    else:
        st.warning("⚠️ No hay ganancia ni pérdida")