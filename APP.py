"""
App de predicción de riesgo de prórroga en contratos SECOP II (Antioquia)
Autor: Antonio José Martínez Anchila

Antes de ejecutar: coloca en la misma carpeta los 6 archivos .joblib
generados por Notebook_Despliegue_SECOP_II.ipynb:
onehot_encoder.joblib, minmax_scaler.joblib, target_encoding_maps.joblib,
random_forest_model.joblib, feature_order.joblib, pipeline_metadata.joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Riesgo de Prórroga SECOP II - Antioquia", page_icon="📄")


@st.cache_resource
def cargar_artefactos():
    return {
        "ohe": joblib.load("onehot_encoder.joblib"),
        "scaler": joblib.load("minmax_scaler.joblib"),
        "te_maps": joblib.load("target_encoding_maps.joblib"),
        "modelo": joblib.load("random_forest_model.joblib"),
        "feature_order": joblib.load("feature_order.joblib"),
        "meta": joblib.load("pipeline_metadata.joblib"),
    }


art = cargar_artefactos()
ohe, scaler, te_maps, modelo = art["ohe"], art["scaler"], art["te_maps"], art["modelo"]
feature_order, meta = art["feature_order"], art["meta"]
onehot_cols, binarias = meta["onehot_cols"], meta["binarias"]
target_enc_cols, numericas = meta["target_enc_cols"], meta["numericas"]

st.title("📄 Riesgo de Prórroga en Contratos SECOP II (Antioquia)")
st.caption(
    "Modelo de clasificación binaria entrenado sobre contratos cerrados firmados "
    "entre 2018 y 2024 (población corregida por sesgo de censura temporal). "
    "Predice si un contrato requerirá una prórroga de tiempo, usando solo "
    "variables conocidas al momento de la firma (t0)."
)

anio_min, anio_max = meta["rango_anio_firma"]
st.info(
    f"⚠️ El modelo solo es válido para contratos con año de firma entre "
    f"**{anio_min} y {anio_max}**. Fuera de ese rango, la predicción no es confiable, "
    f"porque los contratos más recientes aún no han tenido tiempo de mostrar si "
    f"requieren o no una prórroga."
)

st.subheader("Datos del contrato")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Categóricas (One-Hot)**")
    valores_onehot = {}
    for col in onehot_cols:
        opciones = sorted(ohe.categories_[onehot_cols.index(col)].tolist())
        valores_onehot[col] = st.selectbox(col.replace("_", " ").capitalize(), opciones)

    st.markdown("**Alta cardinalidad (Target Encoding)**")
    ciudad = st.selectbox("Ciudad", sorted(meta["ciudades_conocidas"]))
    categoria_segmento = st.selectbox(
        "Categoría UNSPSC (segmento, 6 dígitos)", sorted(meta["categorias_segmento_conocidas"])
    )

with col2:
    st.markdown("**Banderas contractuales (Sí/No)**")
    valores_binarias = {}
    etiquetas_binarias = {
        "es_grupo": "¿Es consorcio o unión temporal?",
        "es_pyme": "¿El contratista es PYME?",
        "habilita_pago_adelantado": "¿Habilita pago adelantado?",
        "obligaci_n_ambiental": "¿Tiene obligación ambiental?",
        "obligaciones_postconsumo": "¿Tiene obligaciones postconsumo?",
        "reversion": "¿Aplica cláusula de reversión?",
        "espostconflicto": "¿Asociado al posconflicto?",
        "el_contrato_puede_ser_prorrogado": "¿El contrato puede ser prorrogado (cláusula)?",
    }
    for col in binarias:
        resp = st.selectbox(etiquetas_binarias.get(col, col), ["No", "Sí"], key=col)
        valores_binarias[col] = 1 if resp == "Sí" else 0

    st.markdown("**Numéricas**")
    valor_contrato = st.number_input(
        "Valor del contrato (COP)", min_value=1000, value=20_000_000, step=100000
    )
    valor_pago_adelantado = st.number_input("Valor de pago adelantado (COP)", min_value=0, value=0)
    pgn = st.number_input("Presupuesto General de la Nación (COP)", min_value=0, value=0)
    sgp = st.number_input("Sistema General de Participaciones (COP)", min_value=0, value=0)
    regalias = st.number_input("Sistema General de Regalías (COP)", min_value=0, value=0)
    credito = st.number_input("Recursos de crédito (COP)", min_value=0, value=0)
    propios = st.number_input("Recursos propios (COP)", min_value=0, value=20_000_000)
    duracion_valor = st.number_input("Duración pactada del contrato (en días)", min_value=1, value=90)
    plazo_pactado_dias = st.number_input("Plazo pactado (días, fecha fin - fecha inicio)", min_value=0, value=90)
    anio_firma = st.number_input("Año de firma", min_value=anio_min, max_value=anio_max, value=anio_max)
    mes_firma = st.number_input("Mes de firma", min_value=1, max_value=12, value=6)

predecir = st.button("Predecir riesgo de prórroga", type="primary")

if predecir:
    fila = {
        **valores_onehot,
        **valores_binarias,
        "ciudad": ciudad,
        "categoria_segmento": categoria_segmento,
        "valor_de_pago_adelantado": valor_pago_adelantado,
        "presupuesto_general_de_la_nacion_pgn": pgn,
        "sistema_general_de_participaciones": sgp,
        "sistema_general_de_regal_as": regalias,
        "recursos_de_credito": credito,
        "recursos_propios": propios,
        "duracion_valor": duracion_valor,
        "plazo_pactado_dias": plazo_pactado_dias,
        "anio_firma": anio_firma,
        "mes_firma": mes_firma,
        "log_valor_contrato": np.log1p(valor_contrato),
    }
    df_new = pd.DataFrame([fila])

    ohe_arr = ohe.transform(df_new[onehot_cols])
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(onehot_cols))

    te_df = pd.DataFrame()
    for col in target_enc_cols:
        te_df[col + "_te"] = df_new[col].map(te_maps[col]).fillna(te_maps["media_global"])

    X_new = pd.concat(
        [df_new[binarias + numericas[:11]].reset_index(drop=True), ohe_df, te_df], axis=1
    )
    X_new[numericas] = scaler.transform(X_new[numericas])
    X_new = X_new.astype(float)

    # Reindexado crítico: alinear exactamente al orden de columnas del entrenamiento
    X_new = X_new.reindex(columns=feature_order, fill_value=0)

    pred = modelo.predict(X_new)[0]
    proba = modelo.predict_proba(X_new)[0][1]

    st.subheader("Resultado")
    if pred == 1:
        st.error(f"⚠️ Riesgo de prórroga (probabilidad estimada: {proba:.1%})")
    else:
        st.success(f"✅ Sin riesgo de prórroga (probabilidad estimada: {proba:.1%})")

    st.caption(
        "Este resultado es una estimación estadística basada en contratos históricos "
        "2018-2024 de Antioquia y no constituye una garantía de ejecución."
    )
