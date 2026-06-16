
C1 = "#cf86b4"
C2 = "#99bdc6"
C3 = "#60ca9a"
C4 = "#87ec9c"
C5 = "#cbf7cf"

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Sistema Predictivo de Ganancia",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# COLORES
# =====================================================

C1 = "#cf86b4"
C2 = "#99bdc6"
C3 = "#60ca9a"
C4 = "#87ec9c"
C5 = "#cbf7cf"

# =====================================================
# MODELO
# =====================================================

modelo = joblib.load("modelo_clf.pkl")
preprocesador = joblib.load("preprocesador_clf.pkl")

# =====================================================
# DATA
# =====================================================

url = "https://docs.google.com/spreadsheets/d/1r0w8oxZ8V2lx1o79jXFOxHwAjq_Sn-jXXj-d1nSLhCo/export?format=csv"

df = pd.read_csv(url)


# =====================================================
# TITULO
# =====================================================

st.title("📊 Sistema Predictivo de Nivel de Ganancia")

st.markdown(
    """
    Ingrese los datos de la venta para estimar el nivel de ganancia.
    """
)

# =====================================================
# INPUTS
# =====================================================

st.subheader("📝 Datos de la Venta")

col1, col2 = st.columns(2)

with col1:

    n_productos = st.number_input(
        "📦 Número de Productos",
        min_value=1,
        value=5
    )

    precio_promedio = st.number_input(
        "💰 Precio Promedio",
        min_value=1.0,
        value=100.0
    )

    desc_maximo = st.slider(
        "🏷️ Descuento Máximo",
        0.0,
        1.0,
        0.10
    )

with col2:

    categoria_principal = st.selectbox(
        "🛍️ Categoría",
        sorted(df["categoria_principal"].dropna().unique())
    )

    channel = st.selectbox(
        "📡 Canal",
        sorted(df["channel"].dropna().unique())
    )

    country = st.selectbox(
        "🌎 País",
        sorted(df["country"].dropna().unique())
    )


# =====================================================
# PREDICCION
# =====================================================

if st.button("🔮 Predecir Nivel de Ganancia"):

    nuevo = pd.DataFrame({

        "n_productos":[n_productos],
        "precio_promedio":[precio_promedio],
        "desc_maximo":[desc_maximo],
        "categoria_principal":[categoria_principal],
        "channel":[channel],
        "country":[country]

    })

    nuevo_proc = preprocesador.transform(nuevo)

    pred = modelo.predict(nuevo_proc)[0]

    prob = modelo.predict_proba(nuevo_proc)[0]

    st.subheader("📊 Resultado de la Predicción")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.info(f"📦 Productos\n\n{n_productos}")

    with t2:
        st.info(f"💰 Precio\n\nS/ {precio_promedio:,.2f}")

    with t3:
        st.info(f"🏷️ Descuento\n\n{desc_maximo*100:.1f}%")

    with t4:

        if pred == "Alta":
            st.success("🟢 Ganancia Alta")

        elif pred == "Media":
            st.warning("🟡 Ganancia Media")

        else:
            st.error("🔴 Ganancia Baja")



    st.divider()

    st.subheader("📊 Probabilidades del Modelo")

    clases = modelo.classes_

    resultado = pd.DataFrame({
        "Clase": clases,
        "Probabilidad": prob
    })

    st.dataframe(
        resultado.style.format({
            "Probabilidad":"{:.2%}"
        }),
        use_container_width=True
    )




    fig, ax = plt.subplots(figsize=(7,4))

    colores = []

    for clase in resultado["Clase"]:

        if clase == "Alta":
            colores.append(C3)

        elif clase == "Media":
            colores.append(C2)

        else:
            colores.append(C1)

    barras = ax.bar(
        resultado["Clase"],
        resultado["Probabilidad"],
        color=colores
    )

    ax.set_title(
        "Probabilidad por Nivel de Ganancia"
    )

    ax.set_ylabel(
        "Probabilidad"
    )

    for barra in barras:

        valor = barra.get_height()

        ax.text(
            barra.get_x()+barra.get_width()/2,
            valor,
            f"{valor:.1%}",
            ha="center"
        )

    st.pyplot(fig)


    fig2, ax2 = plt.subplots(figsize=(5,5))

    ax2.pie(
        resultado["Probabilidad"],
        labels=resultado["Clase"],
        autopct="%1.1f%%",
        colors=[C1, C2, C3]
    )

    ax2.set_title(
        "Distribución de Probabilidades"
    )

    st.pyplot(fig2)



    st.divider()

    st.subheader("📈 Análisis Histórico")

    ventas = (
        df.groupby("categoria_principal")
        .size()
        .sort_values(ascending=False)
        .head(5)
    )

    fig3, ax3 = plt.subplots(figsize=(8,4))

    ax3.bar(
        ventas.index,
        ventas.values,
        color=C2
    )

    ax3.set_title(
        "Top 5 Categorías Más Vendidas"
    )

    plt.xticks(rotation=20)

    st.pyplot(fig3)



    canal = (
        df["channel"]
        .value_counts()
    )

    fig4, ax4 = plt.subplots(figsize=(5,5))

    ax4.pie(
        canal.values,
        labels=canal.index,
        autopct="%1.1f%%",
        colors=[C1, C2, C3, C4, C5]
    )

    ax4.set_title(
        "Distribución por Canal"
    )

    st.pyplot(fig4)



    st.divider()

    if pred == "Alta":

        st.success(
            "🟢 La venta presenta una alta probabilidad de generar ganancias elevadas."
        )

    elif pred == "Media":

        st.warning(
            "🟡 La venta presenta una probabilidad moderada de generar ganancias."
        )

    else:

        st.error(
            "🔴 La venta presenta riesgo de generar una ganancia baja."
        )

# =====================================================
# EJECUTAR
# =====================================================

# streamlit run pagina.py

