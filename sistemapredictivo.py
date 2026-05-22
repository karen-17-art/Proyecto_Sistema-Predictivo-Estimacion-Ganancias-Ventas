import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------
# CARGAR TUS DATOS
# -------------------------
data = pd.read_excel("data_final.xlsx")

# Limpiar datos
data = data.dropna()

# -------------------------
# VARIABLES (IMPORTANTE)
# -------------------------
X = data[['unit_price', 'porcentaje_descuento', 'quantity', 'margen_unitario']]
y = data['ganancia']

# -------------------------
# MODELO
# -------------------------
modelo = LinearRegression()
modelo.fit(X, y)

# -------------------------
# INTERFAZ WEB
# -------------------------
st.title("Sistema Predictivo de Ganancia")

st.write("Ingrese los datos del producto:")

precio = st.number_input("Precio de venta (unit_price)")
descuento = st.number_input("Porcentaje de descuento")
cantidad = st.number_input("Cantidad")
margen = st.number_input("Margen unitario")

if st.button("Predecir ganancia"):
    entrada = pd.DataFrame({
        'unit_price': [precio],
        'porcentaje_descuento': [descuento],
        'quantity': [cantidad],
        'margen_unitario': [margen]
    })

    prediccion = modelo.predict(entrada)

    st.success(f"Ganancia estimada: S/ {prediccion[0]:.2f}")