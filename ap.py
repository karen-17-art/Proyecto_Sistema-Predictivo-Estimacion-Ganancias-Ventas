# =====================================================
# LIBRERÍAS
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Sistema Predictivo",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# COLORES EMPRESARIALES
# =====================================================

C1 = "#cf86b4"
C2 = "#99bdc6"
C3 = "#60ca9a"
C4 = "#87ec9c"
C5 = "#cbf7cf"

# =====================================================
# TÍTULO
# =====================================================

st.title("📊 Sistema Predictivo de Ganancia")

# =====================================================
# DATA
# =====================================================

url = "https://docs.google.com/spreadsheets/d/1RwKQhQbqvDNd3xJt5wiI6R4C7hAoc3TCBNWH3HcHO8g/export?format=csv"

data = pd.read_csv(url)

# =====================================================
# LIMPIEZA
# =====================================================

cols = [
    "ganancia_total",
    "n_productos",
    "precio_promedio",
    "desc_promedio"
]

for c in cols:

    data[c] = pd.to_numeric(
        data[c],
        errors="coerce"
    )

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna(
    subset=["ganancia_total"]
)

data = data.fillna(0)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

data["ingreso_bruto"] = (
    data["n_productos"]
    * data["precio_promedio"]
)

data["costo_estimado"] = (
    data["ingreso_bruto"]
    * 0.65
)

data["impacto_descuento"] = (
    data["ingreso_bruto"]
    * data["desc_promedio"]
)

data["en_campana"] = data["en_campana"].map({
    "True":1,
    "False":0,
    True:1,
    False:0
}).fillna(0)

data["dia_semana"] = (
    data["dia_semana"]
    .astype(str)
    .map({
        "Monday":1,
        "Tuesday":2,
        "Wednesday":3,
        "Thursday":4,
        "Friday":5,
        "Saturday":6,
        "Sunday":7
    })
    .fillna(0)
)

data["mes"] = pd.to_numeric(
    data["mes"],
    errors="coerce"
).fillna(0)

# =====================================================
# FEATURES
# =====================================================

features = [
    "n_productos",
    "precio_promedio",
    "desc_promedio",
    "costo_estimado",
    "ingreso_bruto",
    "impacto_descuento",
    "en_campana",
    "dia_semana",
    "mes",
    "channel",
    "categoria_principal"
]

X = data[features]

X = pd.get_dummies(
    X,
    columns=[
        "channel",
        "categoria_principal"
    ],
    drop_first=True
)

y = data["ganancia_total"]

# =====================================================
# MODELO
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# =====================================================
# INPUTS
# =====================================================

st.subheader("📝 Datos de la Venta")

col1, col2 = st.columns(2)

with col1:

    n_productos = st.number_input(
        "📦 Productos",
        min_value=1,
        max_value=1000,
        value=5
    )

    precio_promedio = st.number_input(
        "💰 Precio promedio",
        min_value=1.0,
        max_value=10000.0,
        value=100.0
    )

    desc_promedio = st.slider(
        "🏷️ Descuento",
        min_value=0.0,
        max_value=1.0,
        value=0.10
    )

with col2:

    channel = st.selectbox(
        "📡 Canal",
        sorted(
            data["channel"].unique()
        )
    )

    categoria_principal = st.selectbox(
        "🛍️ Categoría",
        sorted(
            data["categoria_principal"].unique()
        )
    )

    en_campana = st.selectbox(
        "🔥 Campaña",
        [0, 1]
    )

# =====================================================
# INPUT DATA
# =====================================================

ingreso_bruto = (
    n_productos *
    precio_promedio
)

costo_estimado = (
    ingreso_bruto *
    0.65
)

impacto_descuento = (
    ingreso_bruto *
    desc_promedio
)

input_data = pd.DataFrame([{

    "n_productos": n_productos,
    "precio_promedio": precio_promedio,
    "desc_promedio": desc_promedio,
    "costo_estimado": costo_estimado,
    "ingreso_bruto": ingreso_bruto,
    "impacto_descuento": impacto_descuento,
    "en_campana": en_campana,
    "dia_semana": 3,
    "mes": 5,
    "channel": channel,
    "categoria_principal": categoria_principal

}])

input_data = pd.get_dummies(
    input_data
)

input_data = input_data.reindex(
    columns=X.columns,
    fill_value=0
)

# =====================================================
# PREDICCIÓN
# =====================================================

if st.button(
    "🔮 Predecir Ganancia",
    key="btn_prediccion"
):

    pred = model.predict(input_data)[0]

    ganancia = max(0, pred)

    descuento = impacto_descuento

    promedio = data["ganancia_total"].mean()

    rentabilidad = (
        (ganancia / ingreso_bruto) * 100
        if ingreso_bruto > 0
        else 0
    )

    # =================================================
    # RESULTADOS
    # =================================================

    st.subheader(
        "📊 Resultado de la Predicción"
    )

    st.divider()

    t1, t2, t3, t4 = st.columns(4)

    with t1:

        if ingreso_bruto >= 3000:
            st.success(
                f"💰 Ingreso\n\nS/ {ingreso_bruto:,.2f}"
            )
        elif ingreso_bruto >= 1500:
            st.warning(
                f"💰 Ingreso\n\nS/ {ingreso_bruto:,.2f}"
            )
        else:
            st.error(
                f"💰 Ingreso\n\nS/ {ingreso_bruto:,.2f}"
            )

    with t2:

        if descuento <= ingreso_bruto * 0.10:
            st.success(
                f"🏷️ Descuento\n\nS/ {descuento:,.2f}"
            )
        elif descuento <= ingreso_bruto * 0.20:
            st.warning(
                f"🏷️ Descuento\n\nS/ {descuento:,.2f}"
            )
        else:
            st.error(
                f"🏷️ Descuento\n\nS/ {descuento:,.2f}"
            )

    with t3:

        if ganancia >= promedio:
            st.success(
                f"📈 Ganancia\n\nS/ {ganancia:,.2f}"
            )
        elif ganancia >= promedio * 0.7:
            st.warning(
                f"📈 Ganancia\n\nS/ {ganancia:,.2f}"
            )
        else:
            st.error(
                f"📈 Ganancia\n\nS/ {ganancia:,.2f}"
            )

    with t4:

        if rentabilidad >= 25:
            st.success(
                f"📊 Rentabilidad\n\n{rentabilidad:.1f}%"
            )
        elif rentabilidad >= 10:
            st.warning(
                f"📊 Rentabilidad\n\n{rentabilidad:.1f}%"
            )
        else:
            st.error(
                f"📊 Rentabilidad\n\n{rentabilidad:.1f}%"
            )

    st.divider()

    st.subheader(
        "📊 Análisis Visual"
    )

    g1, g2, g3 = st.columns(3)

    # =================================================
    # GRÁFICO 1
    # =================================================

    with g1:

        st.subheader("📦 Estructura de la venta")

        fig, ax = plt.subplots(
            figsize=(5,4)
        )

        bars = ax.bar(
            [
                "Ingreso",
                "Descuento",
                "Ganancia"
            ],
            [
                ingreso_bruto,
                descuento,
                ganancia
            ],
            color=[
                "#cf86b4",
                "#99bdc6",
                "#60ca9a"
            ]
        )


        ax.grid(
            axis="y",
            alpha=0.25
        )

        for bar in bars:

            yval = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                yval,
                f"S/ {yval:,.0f}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)

    # =================================================
    # GRÁFICO 2
    # =================================================

    with g2:

        st.subheader(
            "🥧 Impacto Descuento"
        )

        sin_desc = max(
            ingreso_bruto - costo_estimado,
            1
        )

        con_desc = max(
            ingreso_bruto - costo_estimado - descuento,
            1
        )

        fig, ax = plt.subplots(
            figsize=(5,4)
        )

        ax.pie(
            [
                sin_desc,
                con_desc
            ],
            labels=[
                "Sin descuento",
                "Con descuento"
            ],
            autopct="%1.1f%%",
            colors=[
                "#99bdc6",
                "#cbf7cf"
            ]
        )

        

        st.pyplot(fig)

    # =================================================
    # GRÁFICO 3
    # =================================================

    with g3:

        st.subheader(
            "📊 Rendimiento vs Promedio"
        )

        fig, ax = plt.subplots(
            figsize=(5,4)
        )

        ax.pie(
            [
                max(ganancia,1),
                max(promedio,1)
            ],
            labels=[
                "Tu ganancia",
                "Promedio"
            ],
            autopct="%1.1f%%",
            colors=[
                "#cf86b4",
                "#60ca9a"
            ]
        )

        st.pyplot(fig)



    st.divider()

    if ganancia > promedio:

        st.success(
            f"🟢 Excelente. La ganancia estimada de S/ {ganancia:,.2f} está por encima del promedio histórico."
        )

    else:

        st.warning(
            f"🟡 La ganancia estimada de S/ {ganancia:,.2f} está por debajo del promedio histórico."
        )

# =====================================================
# EJECUTAR
# =====================================================

# streamlit run ap.py

