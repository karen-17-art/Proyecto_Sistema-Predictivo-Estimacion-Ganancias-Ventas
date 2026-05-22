# =====================================================
# INSTALACIÓN DE LIBRERÍAS
# =====================================================

# pip install streamlit pandas numpy scikit-learn

# =====================================================
# IMPORTACIONES
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# =====================================================
# ENLACE DE LA DATA
# =====================================================

url = "https://docs.google.com/spreadsheets/d/1KypSiIv_hXmbkUPzkkcg0Zdrz2PrUFXSnJtV6R2JnDg/export?format=csv"

# =====================================================
# CARGAR DATA
# =====================================================

data = pd.read_csv(url)

# =====================================================
# LIMPIEZA DE DATOS
# =====================================================

# Eliminar columnas innecesarias

data = data.drop(columns=[
    "sale_id",
    "item_total",
    "margen_unitario",
    "descuento_campania",
    "discount_type",
    "year"
])

# Mantener descuentos válidos

data = data[
    (data['porcentaje_descuento'] >= 0) &
    (data['porcentaje_descuento'] <= 1)
]

# Winsorización de ganancia

p1 = data['ganancia'].quantile(0.01)
p99 = data['ganancia'].quantile(0.99)

data['ganancia_w'] = np.clip(
    data['ganancia'],
    p1,
    p99
)

# =====================================================
# VARIABLES
# =====================================================

X = data[['porcentaje_descuento', 'quantity', 'category']]

y = data['ganancia_w']

# Convertir categóricas

X = pd.get_dummies(X, drop_first=True)

# =====================================================
# TRAIN TEST
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODELO
# =====================================================

model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# PÁGINA WEB
# =====================================================

st.set_page_config(
    page_title="Predicción de Ganancias",
    layout="centered"
)

st.title("Predicción de Ganancias")

st.write("Ingresa los datos para estimar la ganancia.")

# =====================================================
# INPUTS
# =====================================================

descuento = st.number_input(
    "Porcentaje de Descuento",
    min_value=0.0,
    max_value=0.99,
    value=0.10,
    step=0.01
)

cantidad = st.number_input(
    "Cantidad",
    min_value=1,
    value=10
)

categoria = st.selectbox(
    "Categoría",
    ['SHOES', 'SLEEPWEAR', 'T-SHIRTS', 'DRESSES', 'PANTS']
)

# =====================================================
# INPUT DATA
# =====================================================

input_data = pd.DataFrame({
    'porcentaje_descuento': [descuento],
    'quantity': [cantidad],
    'category': [categoria]
})

# Convertir categóricas

input_data = pd.get_dummies(input_data)

# Igualar columnas

input_data = input_data.reindex(
    columns=X.columns,
    fill_value=0
)

# =====================================================
# PREDICCIÓN
# =====================================================

if st.button("Predecir Ganancia"):

    prediccion = model.predict(input_data)

    st.success(
        f"Ganancia estimada: ${prediccion[0]:,.2f}"
    )