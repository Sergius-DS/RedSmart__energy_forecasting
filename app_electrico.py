import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Pronóstico de Demanda Eléctrica con Skforecast",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("⚡ Pronóstico de la Demanda Eléctrica (MW)")
st.subheader("Modelo XGBoost + skforecast")

# --- Constants ---
LAG_STEPS = 96 # The best lag found (96 half-hours = 2 days)
TIME_STEP_MINUTES = 30
STEPS_PER_DAY = 48
EXCEL_FILE = "DemandaCOES_.xlsx"

# Mapping for day name dummies
DAYS_TRANSLATION = {
    'Monday': '1Lunes',
    'Tuesday': '2Martes',
    'Wednesday': '3Miércoles',
    'Thursday': '4Jueves',
    'Friday': '5Viernes',
    'Saturday': '6Sábado',
    'Sunday': '0Domingo'
}

# 1. Get the sorted translated day names (e.g., '0Domingo', '1Lunes', ...)
all_translated_days = sorted(DAYS_TRANSLATION.values())
# 2. Prepend 'dia_' to get the final dummy column names
all_day_dummy_cols = [f'dia_{d}' for d in all_translated_days]

# Lista de todas as possíveis características exógenas
ALL_FEATURES = ['ciclo', 'feriado'] + all_day_dummy_cols

# 🔴 SOLUÇÃO DEFINITIVA: Forçar a ordenação alfabética, que corresponde à ordem do modelo.
# ['ciclo', 'dia_0Domingo', 'dia_1Lunes', ..., 'dia_6Sábado', 'feriado']
REQUIRED_EXOG_COLS = sorted(ALL_FEATURES)


# --- Feature Engineering Functions ---

def create_exogenous_features(data):
    """
    Creates the 'ciclo', day dummies, and 'feriado' features, 
    y aplica la ordenación requerida (alfabética).
    """
    data_with_features = data.copy()

    # 1. Cyclic Feature
    data_with_features['ciclo'] = data_with_features.index.map(
        lambda t: (t.hour * 60 + t.minute) / (24 * 60)
    )

    # 2. Daily Dummies (One-Hot Encoding)
    data_with_features['dia'] = data_with_features.index.day_name().map(DAYS_TRANSLATION)
    # Ensure all possible day values are known to pandas for consistent dummy creation
    data_with_features['dia'] = pd.Categorical(
        data_with_features['dia'], 
        categories=all_translated_days
    )
    data_with_features = pd.get_dummies(data_with_features, columns=['dia'], dtype=int)

    # 3. Holiday Feature
    start_year = data_with_features.index.min().year - 1
    end_year = data_with_features.index.max().year + 2
    pe = holidays.Peru(years=range(start_year, end_year), observed=True)
    data_with_features['feriado'] = data_with_features.index.normalize().isin(pe).astype(int)

    # Drop the original 'Demand' column
    if 'Demand' in data_with_features.columns:
        exog = data_with_features.drop(columns=['Demand'])
    else:
        exog = data_with_features
        
    # --- CRITICAL FIX IMPLEMENTATION: Column ordering enforcement ---
    
    # 1. Ensure all columns in REQUIRED_EXOG_COLS exist
    for col in REQUIRED_EXOG_COLS:
        if col not in exog.columns:
            exog[col] = 0
            
    # 2. Enforce the required column order (Alphabetical)
    exog = exog[REQUIRED_EXOG_COLS]

    return exog

# --- Data Loading and Preprocessing ---

@st.cache_data(show_spinner="Cargando y pre-procesando datos...")
def load_and_preprocess_data():
    """Loads, cleans, and pre-processes the time series data."""
    try:
        data = pd.read_excel(EXCEL_FILE, skiprows=3)
    except FileNotFoundError:
        st.error(f"Error: Archivo '{EXCEL_FILE}' no encontrado. Asegúrese de que el archivo esté en el mismo directorio.")
        return None, None

    data['FECHA'] = pd.to_datetime(data['FECHA'], format='%d/%m/%Y %H:%M')
    data.set_index('FECHA', inplace=True)
    data.rename(columns={'EJECUTADO': 'Demand'}, inplace=True)
    data = data.asfreq(f"{TIME_STEP_MINUTES}min")

    y = data['Demand'].copy()
    exog = create_exogenous_features(data[['Demand']])

    return y, exog

# --- Model Training (Cached Resource) ---

@st.cache_resource(show_spinner="Entrenando modelo de pronóstico...")
def train_forecaster(y_train, x_train):
    """Trains the final XGBoost forecaster."""
    regressor = XGBRegressor(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        random_state=123,
        n_jobs=-1
    )

    forecaster = ForecasterRecursive(
        regressor=regressor,
        lags=LAG_STEPS
    )

    forecaster.fit(y=y_train, exog=x_train)
    
    # Store the columns used during fit in session_state for later comparison
    st.session_state['fit_exog_cols'] = x_train.columns.tolist()
    st.sidebar.write(f"DEBUG (Fit): Columnas exógenas usadas para entrenar: {st.session_state['fit_exog_cols']}")
    
    return forecaster

# --- Prediction and Metrics ---

def generate_prediction_exog(start_date, steps):
    """Generates a future date range and the required exogenous variables."""
    future_index = pd.date_range(
        start=start_date,
        periods=steps,
        freq=f"{TIME_STEP_MINUTES}min"
    )

    future_data = pd.DataFrame(index=future_index)
    exog_pred = create_exogenous_features(future_data)

    return exog_pred

def calculate_metrics(y_true, y_pred):
    """Calculates and returns key forecasting metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_elements = y_true != 0
        if not np.any(non_zero_elements):
            return np.nan
        return np.mean(np.abs((y_true[non_zero_elements] - y_pred[non_zero_elements]) / y_true[non_zero_elements])) * 100

    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "MSE (Error Cuadrático Medio)": f"{mse:,.2f}",
        "RMSE (Raíz del Error Cuadrático Medio)": f"{rmse:,.2f} MW",
        "MAE (Error Absoluto Medio)": f"{mae:,.2f} MW",
        "MAPE (Error Porcentual Absoluto Medio)": f"{mape:,.2f}%",
        "Precisión (100 - MAPE)": f"{100 - mape:,.2f}%"
    }

# --- Streamlit UI and Logic ---

# 1. Load Data
y, exog = load_and_preprocess_data()
if y is None:
    st.stop()

# 2. Train Model
forecaster = train_forecaster(y, exog)

# --- Sidebar for User Input ---
st.sidebar.header("Parámetros de Pronóstico")

# Prediction Horizon Selection
prediction_mode = st.sidebar.radio(
    "Seleccionar Horizonte de Pronóstico",
    ('Diario (24h)', 'Semanal (7 días)', 'Mensual (30 días)')
)

if prediction_mode == 'Diario (24h)':
    steps = STEPS_PER_DAY
    time_label = "1 Día (48 pasos)"
elif prediction_mode == 'Semanal (7 días)':
    steps = STEPS_PER_DAY * 7
    time_label = "7 Días (336 pasos)"
else: # Mensual
    steps = STEPS_PER_DAY * 30
    time_label = "30 Días (1440 pasos)"

# Prediction Start Date Selection
last_date = y.index[-1]
default_start_date = last_date + pd.Timedelta(minutes=TIME_STEP_MINUTES)

start_date = st.sidebar.date_input(
    "Fecha y Hora de Inicio del Pronóstico",
    value=default_start_date,
    min_value=default_start_date,
    max_value=default_start_date + pd.Timedelta(days=365*2)
)

start_datetime = pd.to_datetime(start_date) + pd.Timedelta(
    hours=default_start_date.hour,
    minutes=default_start_date.minute
)

st.sidebar.markdown(f"**Pronóstico:** {time_label}")
st.sidebar.markdown(f"**Inicio:** {start_datetime.strftime('%Y-%m-%d %H:%M')}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo:** XGBoost")
st.sidebar.markdown(f"**Lags (Memoria):** {LAG_STEPS} ({LAG_STEPS/STEPS_PER_DAY} días)")

# --- Main Page Content ---

if st.button(f"Generar Pronóstico para {time_label}"):
    with st.spinner(f"Calculando pronóstico de {time_label}..."):

        # 1. Generate Exogenous Variables for the Future Period
        exog_pred = generate_prediction_exog(start_datetime, steps)

        # --- DEBUGGING LINES (using stored fit_exog_cols) ---
        fit_cols_list = st.session_state.get('fit_exog_cols', [])
        predict_cols_list = exog_pred.columns.tolist()

        st.sidebar.write(f"DEBUG (Predict): Columnas exógenas para la predicción: {predict_cols_list}")

        # Se espera que agora a lista seja idêntica
        if fit_cols_list != predict_cols_list:
            st.error("❌ ERRO: Foi detectada uma discrepância nas colunas exógenas ANTES da previsão!")
            st.error(f"Colunas usadas durante o TREINAMENTO: {fit_cols_list}")
            st.error(f"Colunas geradas para a PREVISÃO: {predict_cols_list}")
            st.warning("A ordem e os nomes DEVEM ser IDÊNTICOS. Isso não deveria acontecer agora. Verifique a ordem na função create_exogenous_features.")
            st.stop()
        else:
            st.sidebar.write("✅ DEBUG: A verificação manual de colunas exógenas passou. Nomes e ordem são idênticos.")

        # 2. Make Prediction (Agora deve funcionar com a ordem correta)
        predictions = forecaster.predict(steps=steps, exog=exog_pred)

        # 3. Display Results
        st.header(f"Resultados del Pronóstico: {prediction_mode}")

        # --- Chart ---
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Gráfica de la Demanda Eléctrica")
            fig, ax = plt.subplots(figsize=(14, 6))

            # Plot recent historical data for context (e.g., last 30 days)
            context_steps = STEPS_PER_DAY * 30
            y_context = y[-context_steps:]
            y_context.plot(ax=ax, label='Demanda Histórica (MW)', color='gray', alpha=0.7)

            # Plot the forecast
            predictions.plot(ax=ax, label=f'Pronóstico {prediction_mode} (MW)', color='red', linestyle='--')

            ax.set_title(f'Pronóstico de Demanda Eléctrica: {start_datetime.strftime("%Y-%m-%d")} a {predictions.index[-1].strftime("%Y-%m-%d")}')
            ax.set_xlabel('Fecha y Hora')
            ax.set_ylabel('Demanda (MW)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # --- Metrics ---
        with col2:
            st.subheader("Información Clave")
            st.markdown(f"**Inicio del Pronóstico:** {predictions.index[0].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Fin del Pronóstico:** {predictions.index[-1].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Demanda Promedio Pronosticada:** **{predictions.mean():,.2f} MW**")
            st.markdown(f"**Pico Máximo Pronosticado:** **{predictions.max():,.2f} MW**")
            st.markdown(f"**Valle Mínimo Pronosticado:** **{predictions.min():,.2f} MW**")


        # --- Metric Note (Since we don't have a true test set for *this* prediction) ---
        st.markdown(
            """
            ⚠️ **Nota sobre Métricas:** El modelo se entrenó con todos los datos históricos disponibles.
            Las métricas de error del script original (`RMSE: 94.54 MW`, `MAPE: 1.54%` para la última semana del dataset) 
            indican la precisión histórica del modelo sobre datos de prueba.
            """
        )

        # --- Raw Data Table ---
        st.subheader("Tabla de Pronóstico")
        predictions_df = predictions.to_frame(name='Demanda Pronosticada (MW)')
        predictions_df.index.name = 'Fecha y Hora'
        st.dataframe(predictions_df.style.format("{:,.2f}"))

        # Option to download the forecast
        csv = predictions_df.to_csv().encode('utf-8')
        st.download_button(
            label="Descargar Pronóstico como CSV",
            data=csv,
            file_name=f'pronostico_demanda_electrica_{prediction_mode.lower().split(" ")[0]}.csv',
            mime='text/csv',
        )

# --- Historical Performance Section ---

st.header("Análisis de Rendimiento Histórico")

# Re-run backtesting metrics from your script for display
steps_test = 48 * 7
y_test = y[-steps_test:]
x_test = exog[-steps_test:]
y_train = y[:-steps_test]
x_train = exog[:-steps_test]

@st.cache_resource(show_spinner="Calculando rendimiento histórico...")
def get_historical_predictions(y_t, x_t, y_test_data, x_test_data):
    regressor_hist = XGBRegressor(n_estimators=250, max_depth=8, learning_rate=0.05, random_state=123, n_jobs=-1)
    forecaster_hist = ForecasterRecursive(regressor=regressor_hist, lags=LAG_STEPS)
    forecaster_hist.fit(y=y_t, exog=x_t)
    # Store history fit columns for debugging purposes as well
    st.session_state['hist_fit_exog_cols'] = x_t.columns.tolist()  
    return forecaster_hist.predict(steps=len(y_test_data), exog=x_test_data)

predictions_hist = get_historical_predictions(y_train, x_train, y_test, x_test)

# Debugging for historical prediction columns as well
st.sidebar.write(f"DEBUG (Hist Fit): Columnas exógenas usadas para entrenar histórico: {st.session_state.get('hist_fit_exog_cols', [])}")
st.sidebar.write(f"DEBUG (Hist Predict): Columnas exógenas para la predicción histórica: {x_test.columns.tolist()}")


metrics = calculate_metrics(y_test, predictions_hist)

# Display historical metrics in a cleaner format
st.subheader("Métricas de Precisión sobre la Última Semana de Datos (Test Set)")
cols_metrics = st.columns(3)
metrics_list = list(metrics.items())

for i, (name, value) in enumerate(metrics_list):
    cols_metrics[i % 3].metric(label=name, value=value)


# Display the historical plot
st.subheader("Gráfica de Ajuste Histórico (Última Semana)")
fig_hist, ax_hist = plt.subplots(figsize=(14, 6))
y_test.plot(ax=ax_hist, label='Valor Real (MW)', color='blue')
predictions_hist.plot(ax=ax_hist, label='Predicción del Modelo (MW)', color='red', linestyle='--')
ax_hist.set_title('Rendimiento del Modelo en el Conjunto de Prueba')
ax_hist.set_xlabel('Fecha y Hora')
ax_hist.set_ylabel('Demanda (MW)')
ax_hist.legend()
ax_hist.grid(True)
st.pyplot(fig_hist)






