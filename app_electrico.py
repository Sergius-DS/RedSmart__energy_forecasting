import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys # Para la versión de Python
import platform # Para la información del sistema operativo

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

# 🔴 SOLUCIÓN DEFINITIVA: Forzar la ordenación alfabética, que corresponde à ordem do modelo.
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

    # Try to get holidays from the package
    pe_holidays_from_package = {}
    try:
        pe_holidays_from_package = holidays.Peru(years=range(start_year, end_year + 1), observed=True)
    except Exception as e:
        st.warning(f"Error loading holidays from package: {e}. Using fallback for some dates.")
        pe_holidays_from_package = {} # Ensure it's an empty dict if error

    # Fallback: manually define known fixed holidays (as dates)
    # This also handles cases where the package might not have data for far future years
    fixed_holidays_dates = []
    for year in range(start_year, end_year + 1):
        fixed_holidays_dates.extend([
            pd.to_datetime(f"{year}-01-01").date(),   # Año Nuevo
            pd.to_datetime(f"{year}-05-01").date(),   # Día del Trabajo
            pd.to_datetime(f"{year}-07-28").date(),   # Fiestas Patrias - Independencia
            pd.to_datetime(f"{year}-07-29").date(),   # Fiestas Patrias - Batalla de Ayacucho
            pd.to_datetime(f"{year}-08-30").date(),   # Santa Rosa de Lima
            pd.to_datetime(f"{year}-10-08").date(),   # Combate de Angamos
            pd.to_datetime(f"{year}-11-01").date(),   # Día de Todos los Santos
            pd.to_datetime(f"{year}-12-08").date(),   # Inmaculada Concepción
            pd.to_datetime(f"{year}-12-25").date(),   # Navidad
        ])
    
    # Combine holidays from package and fixed holidays
    known_holidays = set(pe_holidays_from_package.keys()) if isinstance(pe_holidays_from_package, dict) else set(pe_holidays_from_package)
    known_holidays.update(fixed_holidays_dates)

    # Create 'feriado' column based on combined holidays
    data_with_features['feriado'] = data_with_features.index.normalize().isin(known_holidays).astype(int)

    # Drop the original 'Demand' column
    if 'Demand' in data_with_features.columns:
        exog = data_with_features.drop(columns=['Demand'])
    else:
        exog = data_with_features

    # --- CRITICAL FIX: Column ordering enforcement ---
    for col in REQUIRED_EXOG_COLS:
        if col not in exog.columns:
            exog[col] = 0

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

    # If x_train is empty, skforecast might implicitly assume no exogenous variables.
    # We explicitly handle this for the main forecaster as well for consistency.
    if not x_train.empty:
        forecaster.fit(y=y_train, exog=x_train)
        st.session_state['fit_exog_cols'] = x_train.columns.tolist()
        st.session_state['fit_exog_dtypes'] = x_train.dtypes.to_dict()
        st.sidebar.write(f"DEBUG (Fit): Columnas exógenas usadas para entrenar: {st.session_state['fit_exog_cols']}")
        st.sidebar.write(f"DEBUG (Fit): dtypes de las columnas de entrenamiento: {st.session_state['fit_exog_dtypes']}")
    else:
        # If x_train is empty, fit without exog, and store empty lists/dicts
        forecaster.fit(y=y_train) # Fit without exogenous variables
        st.session_state['fit_exog_cols'] = []
        st.session_state['fit_exog_dtypes'] = {}
        st.sidebar.write("DEBUG (Fit): x_train está vacío. El forecaster fue entrenado SIN variables exógenas.")


    # Added for more detailed version info
    try:
        import skforecast
        st.session_state['skforecast_version'] = skforecast.__version__
    except AttributeError:
        st.session_state['skforecast_version'] = "N/A (check pip list)"
    try:
        import holidays as h_lib # avoid name collision
        st.session_state['holidays_version'] = h_lib.__version__
    except AttributeError:
        st.session_state['holidays_version'] = "N/A (check pip list)"

    st.session_state['pandas_version'] = pd.__version__
    st.session_state['python_version'] = sys.version
    st.session_state['platform_system'] = platform.system()

    st.sidebar.write(f"DEBUG (Versions): skforecast: {st.session_state['skforecast_version']}, pandas: {st.session_state['pandas_version']}, holidays: {st.session_state['holidays_version']}, Python: {st.session_state['python_version']}, OS: {st.session_state['platform_system']}")

    return forecaster

# --- Prediction and Metrics ---

def generate_prediction_exog(start_date, steps):
    """Generates a future date range and the required exogenous variables."""
    future_index = pd.date_range(
        start=start_date,
        periods=steps,
        freq=f"{TIME_STEP_MINUTES}min"
    )

    # Ensure no duplicates or gaps in the generated index
    future_index = future_index.drop_duplicates()

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

# Add a debug message about data length
st.sidebar.write(f"DEBUG: Longitud de datos y: {len(y)}, exog: {len(exog)}")


# 2. Train Model
forecaster = train_forecaster(y, exog)

# --- Sidebar for User Input ---
st.sidebar.header("Parámetros de Pronóstico")

# Define the actual start of prediction required by skforecast (one step after y ends)
skforecast_predict_start_datetime = y.index[-1] + pd.Timedelta(minutes=TIME_STEP_MINUTES)

# Prediction Horizon Selection
prediction_mode = st.sidebar.radio(
    "Seleccionar Horizonte de Pronóstico",
    ('Diario (24h)', 'Semanal (7 días)', 'Mensual (30 días)')
)

# `user_horizon_steps` here defines the length of the *user's requested horizon*, not total steps for skforecast
if prediction_mode == 'Diario (24h)':
    user_horizon_steps = STEPS_PER_DAY
    time_label = "1 Día (48 pasos)"
elif prediction_mode == 'Semanal (7 días)':
    user_horizon_steps = STEPS_PER_DAY * 7
    time_label = "7 Días (336 pasos)"
else: # Mensual
    user_horizon_steps = STEPS_PER_DAY * 30
    time_label = "30 Días (1440 pasos)"

# Prediction Start Date Selection
# The min_value for the date input should allow selecting from the actual start of prediction onwards
user_selected_date = st.sidebar.date_input(
    "Fecha de Inicio del Pronóstico (Visualización)",
    value=skforecast_predict_start_datetime, # Default to the actual next step
    min_value=skforecast_predict_start_datetime,
    max_value=skforecast_predict_start_datetime + pd.Timedelta(days=365*2)
)

# Combine user selected date with the *time component of the skforecast_predict_start_datetime*
# This ensures consistency of the time step
user_display_start_datetime = pd.to_datetime(user_selected_date) + pd.Timedelta(
    hours=skforecast_predict_start_datetime.hour,
    minutes=skforecast_predict_start_datetime.minute
)

st.sidebar.markdown(f"**Pronóstico:** {time_label}")
st.sidebar.markdown(f"**Inicio (visualizado):** {user_display_start_datetime.strftime('%Y-%m-%d %H:%M')}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo:** XGBoost")
st.sidebar.markdown(f"**Lags (Memoria):** {LAG_STEPS} ({LAG_STEPS/STEPS_PER_DAY} días)")

# --- Main Page Content ---

if st.button(f"Generar Pronóstico para {time_label}"):
    with st.spinner(f"Calculando pronóstico de {time_label}..."):

        # 🔴 RECALCULATE STEPS FOR SKFORECAST
        # Calculate the total duration from skforecast's actual start to the end of user's requested horizon
        end_of_user_horizon_datetime = user_display_start_datetime + pd.Timedelta(minutes=TIME_STEP_MINUTES * (user_horizon_steps - 1))

        # Ensure prediction starts at least from skforecast_predict_start_datetime
        if end_of_user_horizon_datetime < skforecast_predict_start_datetime:
            st.error("Error: La fecha de finalización del pronóstico es anterior a la fecha de inicio requerida por el modelo.")
            st.stop()

        # Calculate total steps needed for skforecast, starting from skforecast_predict_start_datetime
        # and ending at end_of_user_horizon_datetime
        time_diff = end_of_user_horizon_datetime - skforecast_predict_start_datetime
        total_steps_for_skforecast_predict = int(time_diff.total_seconds() / (TIME_STEP_MINUTES * 60)) + 1

        # Adjust total_steps_for_skforecast_predict if user_display_start_datetime is later than skforecast_predict_start_datetime
        # This will make skforecast predict the "gap" and the user's requested horizon
        
        st.sidebar.write(f"DEBUG: skforecast will predict from {skforecast_predict_start_datetime.strftime('%Y-%m-%d %H:%M')}")
        st.sidebar.write(f"DEBUG: Total steps for forecaster.predict(): {total_steps_for_skforecast_predict}")
        st.sidebar.write(f"DEBUG: User selected forecast display start: {user_display_start_datetime.strftime('%Y-%m-%d %H:%M')}")
        st.sidebar.write(f"DEBUG: User selected forecast horizon steps: {user_horizon_steps}")

        # 1. Generate Exogenous Variables for the FULL period required by skforecast
        exog_pred_full = generate_prediction_exog(skforecast_predict_start_datetime, total_steps_for_skforecast_predict)


        # --- DEBUGGING LINES (using stored fit_exog_cols and dtypes) ---
        fit_cols_list = st.session_state.get('fit_exog_cols', [])
        fit_dtypes_dict = st.session_state.get('fit_exog_dtypes', {})
        predict_cols_list = exog_pred_full.columns.tolist() # Use exog_pred_full for checks
        predict_dtypes_dict = exog_pred_full.dtypes.to_dict()

        st.sidebar.write(f"DEBUG (Predict): Columnas exógenas para la predicción (full): {predict_cols_list}")
        st.sidebar.write(f"DEBUG (Predict): dtypes de las columnas de predicción (full): {predict_dtypes_dict}")
        st.sidebar.write(f"DEBUG (Versions): skforecast: {st.session_state.get('skforecast_version')}, pandas: {st.session_state.get('pandas_version')}, holidays: {st.session_state.get('holidays_version')}, Python: {st.session_state.get('python_version')}, OS: {st.session_state.get('platform_system')}")

        # Additional debug for 'feriado' column values in prediction exog
        if 'feriado' in exog_pred_full.columns:
            st.sidebar.write(f"DEBUG (Predict): feriado values in exog_pred_full: {exog_pred_full['feriado'].value_counts().to_dict()}")
            # Only show sample if there are holidays detected
            if 1 in exog_pred_full['feriado'].value_counts():
                st.sidebar.write(f"DEBUG (Predict): Sample dates with feriado=1:\n{exog_pred_full[exog_pred_full['feriado']==1].index[:5].strftime('%Y-%m-%d %H:%M').tolist()}")


        # Final Safety Check before prediction
        if exog_pred_full.isnull().any().any():
            st.error("❌ ERRO: `exog_pred_full` contiene valores nulos. Revise la generación de características.")
            st.write(exog_pred_full.isnull().sum())
            st.stop()

        if not exog_pred_full.index.is_monotonic_increasing:
            st.error("❌ ERRO: El índice de `exog_pred_full` no está ordenado monótonamente creciente. Esto es crítico.")
            st.stop()

        if exog_pred_full.index.duplicated().any():
            st.error("❌ ERRO: El índice de `exog_pred_full` contiene duplicados. Asegúrese de que la frecuencia sea regular.")
            st.stop()

        # Ensure consistency before prediction
        if not fit_cols_list and not predict_cols_list:
            st.sidebar.write("✅ DEBUG: Ambos entrenamientos y predicciones son sin variables exógenas.")
            # No reindexing or reconstruction needed if both are empty
        elif fit_cols_list != predict_cols_list:
            st.error("❌ ERRO: Foi detectada uma discrepância nas COLUNAS exógenas ANTES da previsão!")
            st.error(f"Colunas usadas durante o TREINAMENTO: {fit_cols_list}")
            st.error(f"Colunas geradas para a PREVISÃO: {predict_cols_list}")
            st.warning("A ordem e os nomes DEVEM ser IDÊNTICOS. Verifique la orden en la función create_exogenous_features.")
            st.stop()
        elif fit_dtypes_dict != predict_dtypes_dict:
            st.error("❌ ERRO: Foi detectada uma discrepância nos TIPOS DE DADOS (dtypes) das colunas exógenas ANTES da previsão!")
            st.error(f"Dtypes usados durante o TREINAMENTO: {fit_dtypes_dict}")
            st.error(f"Dtypes generados para a PREVISÃO: {predict_dtypes_dict}")
            st.warning("Os dtypes deben ser IDÊNTICOS para cada columna.")
            st.stop()
        else:
            st.sidebar.write("✅ DEBUG: A verificação manual de colunas exógenas e seus dtypes passou. Nomes, ordem e dtypes são idénticos.")

            # 🔴 SOLUCIÓN MÁS ROBUSTA: Reconstruir exog_pred_full para asegurar la consistencia total del índice y dtypes de columnas
            st.sidebar.write("DEBUG: Reconstruyendo exog_pred_full para asegurar total consistencia.")
            try:
                reconstructed_exog_pred = pd.DataFrame(
                    0, # Default value, will be overwritten
                    index=exog_pred_full.index,
                    columns=fit_cols_list,
                )

                # Populate with actual values from the generated exog_pred_full
                for col in fit_cols_list:
                    if col in exog_pred_full.columns:
                        reconstructed_exog_pred[col] = exog_pred_full[col]
                
                # Enforce dtypes
                for col, dtype in fit_dtypes_dict.items():
                    if col in reconstructed_exog_pred.columns:
                        reconstructed_exog_pred[col] = reconstructed_exog_pred[col].astype(dtype)
                
                exog_pred_full = reconstructed_exog_pred # Replace the original exog_pred_full
                st.sidebar.write("DEBUG: Reconstrucción y ajuste de dtypes de exog_pred_full aplicados con éxito.")

            except Exception as e:
                st.error(f"❌ ERRO: Falló la reconstrucción de columnas para la predicción: {e}")
                st.stop()

        # 2. Make Prediction with the FULL exog_pred_full
        if fit_cols_list: # If model was trained with exogenous variables
            predictions_full = forecaster.predict(steps=total_steps_for_skforecast_predict, exog=exog_pred_full)
        else: # If model was trained WITHOUT exogenous variables
            predictions_full = forecaster.predict(steps=total_steps_for_skforecast_predict)

        # 3. Slice the full prediction for display purposes, starting from user_display_start_datetime
        predictions = predictions_full.loc[user_display_start_datetime:].copy()


        # 3. Display Results
        st.header(f"Resultados del Pronóstico: {prediction_mode}")

        # --- Chart ---
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Gráfica de la Demanda Eléctrica")
            fig, ax = plt.subplots(figsize=(14, 6))

            # 🔴 CORRECCIÓN AQUÍ: Plot historical data up to the start of the user's requested forecast period
            # This ensures no gap between historical context and the forecast plot.
            context_end_for_plot = user_display_start_datetime - pd.Timedelta(minutes=TIME_STEP_MINUTES)
            
            # Show a reasonable window of historical context, e.g., last 30 days before forecast starts
            # Ensure it doesn't go before the actual start of y
            context_start_for_plot = max(y.index.min(), context_end_for_plot - pd.Timedelta(days=30))
            
            y_context_for_plot = y.loc[context_start_for_plot : context_end_for_plot]

            if not y_context_for_plot.empty:
                y_context_for_plot.plot(ax=ax, label='Demanda Histórica (MW)', color='gray', alpha=0.7)
            else:
                st.warning("No hay suficientes datos históricos para mostrar el contexto en el gráfico principal.")


            # Plot the forecast
            if not predictions.empty:
                predictions.plot(ax=ax, label=f'Pronóstico {prediction_mode} (MW)', color='red', linestyle='--')
                ax.set_title(f'Pronóstico de Demanda Eléctrica: {user_display_start_datetime.strftime("%Y-%m-%d %H:%M")} a {predictions.index[-1].strftime("%Y-%m-%d %H:%M")}')
            else:
                st.warning("No se generaron predicciones para el periodo solicitado.")
                ax.set_title(f'Pronóstico de Demanda Eléctrica: No se generaron predicciones')

            ax.set_xlabel('Fecha y Hora')
            ax.set_ylabel('Demanda (MW)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # --- Metrics ---
        with col2:
            st.subheader("Información Clave")
            if not predictions.empty:
                st.markdown(f"**Inicio del Pronóstico:** {predictions.index[0].strftime('%Y-%m-%d %H:%M')}")
                st.markdown(f"**Fin del Pronóstico:** {predictions.index[-1].strftime('%Y-%m-%d %H:%M')}")
                st.markdown(f"**Demanda Promedio Pronosticada:** **{predictions.mean():,.2f} MW**")
                st.markdown(f"**Pico Máximo Pronosticado:** **{predictions.max():,.2f} MW**")
                st.markdown(f"**Valle Mínimo Pronosticado:** **{predictions.min():,.2f} MW**")
            else:
                st.markdown("No hay datos de pronóstico para mostrar.")


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
        if not predictions_df.empty:
            st.dataframe(predictions_df.style.format("{:,.2f}"))

            # Option to download the forecast
            csv = predictions_df.to_csv().encode('utf-8')
            st.download_button(
                label="Descargar Pronóstico como CSV",
                data=csv,
                file_name=f'pronostico_demanda_electrica_{prediction_mode.lower().split(" ")[0]}.csv',
                mime='text/csv',
            )
        else:
            st.warning("No hay pronósticos para mostrar o descargar.")

# --- Historical Performance Section ---

st.header("Análisis de Rendimiento Histórico (Evaluación del Modelo)")
st.markdown(
    """
    Esta sección muestra el rendimiento del modelo sobre la **última semana de datos históricos disponibles**
    que no fueron utilizados para el entrenamiento final. No está directamente relacionada con la "Fecha de Inicio del Pronóstico"
    seleccionada arriba, que es para la proyección futura. Sirve para entender la precisión del modelo en datos que no "vio" durante su ajuste.
    """
)

# Re-run backtesting metrics from your script for display
steps_test = 48 * 7
# Check if enough data exists for historical performance calculation
# Need enough data for lags + training set + test set
# min_data_needed = LAG_STEPS (for first prediction) + steps_test (for test set) + 1 (at least one point for train)
min_data_needed = LAG_STEPS + steps_test + 1

if len(y) < min_data_needed:
    st.warning(f"No hay suficientes datos históricos (mínimo {min_data_needed} puntos) para calcular el rendimiento histórico sobre la última semana ({steps_test} pasos) más los lags requeridos ({LAG_STEPS} pasos).")
    st.markdown("No se pudo calcular el rendimiento histórico. Asegúrese de que el archivo `DemandaCOES_.xlsx` contenga al menos 7 días de datos para el test set + 2 días para los lags + datos para el train set.")
else:
    y_test = y[-steps_test:]
    x_test = exog[-steps_test:]
    y_train = y[:-steps_test]
    x_train = exog[:-steps_test]

    @st.cache_resource(show_spinner="Calculando rendimiento histórico...")
    def get_historical_predictions(y_t, x_t, y_test_data, x_test_data):
        regressor_hist = XGBRegressor(n_estimators=250, max_depth=8, learning_rate=0.05, random_state=123, n_jobs=-1)
        forecaster_hist = ForecasterRecursive(regressor=regressor_hist, lags=LAG_STEPS)

        # Handle empty x_t (x_train for historical)
        if not x_t.empty:
            forecaster_hist.fit(y=y_t, exog=x_t)
            st.session_state['hist_fit_exog_cols'] = x_t.columns.tolist()
            st.session_state['hist_fit_exog_dtypes'] = x_t.dtypes.to_dict()
            st.sidebar.write(f"DEBUG (Hist Fit): Columnas exógenas usadas para entrenar histórico: {st.session_state.get('hist_fit_exog_cols', [])}")
            st.sidebar.write(f"DEBUG (Hist Fit): dtypes de las columnas de entrenamiento histórico: {st.session_state.get('hist_fit_exog_dtypes', {})}")
        else:
            forecaster_hist.fit(y=y_t)
            st.session_state['hist_fit_exog_cols'] = []
            st.session_state['hist_fit_exog_dtypes'] = {}
            st.sidebar.write("DEBUG (Hist Fit): x_train (histórico) está vacío. El forecaster fue entrenado SIN variables exógenas.")


        # Apply reconstruction to x_test_data before prediction in historical context too
        hist_fit_cols_list = st.session_state['hist_fit_exog_cols']
        hist_fit_dtypes_dict = st.session_state['hist_fit_exog_dtypes']

        st.sidebar.write("DEBUG (Hist): Aplicando reconstrucción forzada de columnas para x_test_data.")
        try:
            # Conditionally reconstruct and adjust dtypes if the historical model was trained with exog
            if hist_fit_cols_list: # Only apply if columns were present during fit
                reconstructed_x_test_data = pd.DataFrame(
                    0, # Default value
                    index=x_test_data.index,
                    columns=hist_fit_cols_list,
                )
                for col in hist_fit_cols_list:
                    if col in x_test_data.columns:
                        reconstructed_x_test_data[col] = x_test_data[col]
                
                for col, dtype in hist_fit_dtypes_dict.items():
                    if col in reconstructed_x_test_data.columns:
                        reconstructed_x_test_data[col] = reconstructed_x_test_data[col].astype(dtype)
                
                x_test_data = reconstructed_x_test_data # Replace the original x_test_data
                st.sidebar.write("DEBUG (Hist): Reconstrucción y ajuste de dtypes aplicados con éxito para x_test_data.")
            else:
                # If historical model was trained without exog, ensure x_test_data for predict is also empty or not passed
                x_test_data = pd.DataFrame(index=x_test_data.index) # Create an empty DataFrame with the correct index
                st.sidebar.write("DEBUG (Hist): El modelo histórico fue entrenado sin exógenas. x_test_data para predicción se ha vaciado.")

        except Exception as e:
            st.error(f"❌ ERRO (Histórico): Falló la reconstrucción de columnas para x_test_data: {e}")
            st.stop()

        # Debugging for historical prediction columns and dtypes as well
        st.sidebar.write(f"DEBUG (Hist Predict): Columnas exógenas para la predicción histórica: {x_test_data.columns.tolist()}")
        st.sidebar.write(f"DEBUG (Hist Predict): dtypes de las columnas de predicción histórica: {x_test_data.dtypes.to_dict()}")

        # Conditionally pass exog to predict based on whether it was used during fit
        if hist_fit_cols_list:
            predictions_hist = forecaster_hist.predict(steps=len(y_test_data), exog=x_test_data)
        else:
            predictions_hist = forecaster_hist.predict(steps=len(y_test_data))

        return predictions_hist

    predictions_hist = get_historical_predictions(y_train, x_train, y_test, x_test)


    if not predictions_hist.empty:
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
        
        # Ensure y_test is not empty before plotting
        if not y_test.empty:
            y_test.plot(ax=ax_hist, label='Valor Real (MW)', color='blue')
        else:
            st.warning("No hay datos reales para mostrar en la gráfica histórica.")

        predictions_hist.plot(ax=ax_hist, label='Predicción del Modelo (MW)', color='red', linestyle='--')
        
        # 🔴 CORRECCIÓN AQUÍ: Título más descriptivo para el histórico
        ax_hist.set_title(f'Rendimiento del Modelo sobre la Última Semana Histórica ({y_test.index.min().strftime("%Y-%m-%d %H:%M")} a {y_test.index.max().strftime("%Y-%m-%d %H:%M")})')
        ax_hist.set_xlabel('Fecha y Hora')
        ax_hist.set_ylabel('Demanda (MW)')
        ax_hist.legend()
        ax_hist.grid(True)
        st.pyplot(fig_hist)
    else:
        st.warning("No se pudo calcular el rendimiento histórico o generar la gráfica debido a la insuficiencia de datos o errores en el procesamiento.")












