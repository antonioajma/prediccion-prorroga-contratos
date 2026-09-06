Predicción de Riesgo de Prórroga en Contratos Públicos — SECOP II Antioquia
App de clasificación binaria que estima la probabilidad de que un contrato público (nuevo o en ejecución) requiera una prórroga de tiempo, usando solo variables conocidas al momento de la firma.
App en producción: https://prediccion-prorroga-contratos.streamlit.app/
Contenido del repositorio
Archivo	Descripción
`APP.py`	Aplicación Streamlit
`requirements.txt`	Dependencias de Python
`random_forest_model.joblib`	Modelo entrenado (Random Forest)
`onehot_encoder.joblib`	Codificador de variables categóricas
`minmax_scaler.joblib`	Escalador de variables numéricas
`target_encoding_maps.joblib`	Diccionarios de Target Encoding (ciudad, categoría UNSPSC)
`feature_order.joblib`	Orden de columnas esperado por el modelo
`pipeline_metadata.joblib`	Metadatos del pipeline (categorías conocidas, listas de columnas)
Modelo
Algoritmo: Random Forest (`n_estimators=200, max_depth=None, min_samples_leaf=2`)
Población de entrenamiento: contratos cerrados en Antioquia firmados entre 2018-2024 (642 positivos reales + 1,800 negativos muestreados)
Desempeño en test: Accuracy 92.6%, F1-score 84.7% en la clase de prórroga
Ejecución local
```bash
pip install -r requirements.txt
streamlit run APP.py
```
Autor
Antonio José Martínez Anchila
