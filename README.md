# SmartEnergyForecasting
## 📊 Pronóstico de Demanda Eléctrica con Skforecast

Este proyecto implementa un flujo completo de **forecasting de series temporales** para la demanda eléctrica en Perú, utilizando la librería [**skforecast**](https://skforecast.org/). El objetivo es comparar un modelo **univariado** (solo lags de la serie) contra un modelo **multivariado** (lags + variables exógenas de calendario).

## 🤖 Despliegue del proyecto 

Se creó una aplicación es un sistema completo de pronóstico de demanda eléctrica alojado en un dashboard de Streamlit.

El núcleo del sistema es un modelo avanzado de Machine Learning (XGBoost) que, utilizando la librería skforecast, realiza predicciones recursivas de alta precisión. La clave de su exactitud reside en una ingeniería de características inteligente que inyecta información contextual al modelo, como los patrones del ciclo diario, el día de la semana y el impacto de los feriados nacionales.

Ofrece una Experiencia de Usuario (UX) totalmente intuitiva que permite configurar el horizonte de pronóstico (de 1 día a 1 mes) sin necesidad de código. Finalmente, proporciona una triple validación de resultados mediante: gráficos interactivos (Plotly), estadísticas operativas (pico, valle, promedio) y una tabla de datos descargable, convirtiendo los datos brutos en inteligencia de negocio lista para la planificación y toma de decisiones.

- **Modelo de Machine Learning:** [Deploy](https://redsmartenergyforecasting-bzadnqof3sjqbbya7dqiuc.streamlit.app//)

## 🚀 Flujo del proyecto

1. **Carga y preparación de datos**

   * Lectura desde Excel (`pandas`).
   * Conversión a `datetime` y fijar frecuencia de 30 minutos.

2. **Análisis exploratorio**

   * Visualización de la serie.
   * Medias móviles (24h, 7d) para identificar tendencias y estacionalidad.

3. **Ingeniería de características**

   * Ciclo intradía (0–1).
   * Variables dummy para día de la semana.
   * Indicador de feriados (librería `holidays`).

4. **Partición de datos**

   * Última semana reservada como conjunto de prueba.

5. **Modelado con Skforecast**

   * **Univariado:** lags de la serie + XGBoost.
   * **Multivariado:** lags + exógenas.
   * Validación temporal con `TimeSeriesFold`.
   * Optimización de hiperparámetros con `grid_search_forecaster`.

6. **Evaluación**

   * Predicciones sobre la última semana.
   * Gráficos comparativos (train, test, predicciones).
   * Métricas: **MSE** y **MAE**.

## 📌 Requisitos

* Python 3.9+
* Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `skforecast`, `xgboost`, `lightgbm`, `scikit-learn`, `holidays`.

## ▶️ Uso

1. Instalar dependencias:

   ```bash
   pip install -q skforecast xgboost lightgbm scikit-learn pandas matplotlib seaborn holidays
   ```
2. Ejecutar el script paso a paso (o el notebook).
3. Revisar los gráficos y métricas para comparar modelos.

## 📈 Resultados esperados

* El modelo **univariado** captura la inercia de la serie.
* El modelo **multivariado** mejora la predicción al incorporar información de calendario y feriados.
