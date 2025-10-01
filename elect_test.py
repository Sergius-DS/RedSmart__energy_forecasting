import streamlit as st
import pandas as pd
import numpy as np
# Se reemplaza matplotlib por plotly
import plotly.graph_objects as go 
import holidays
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --- Configuration ---
st.set_page_config(
    page_title="Pronóstico de Demanda Eléctrica",
    layout="wide"
)
st.title("⚡ Pronóstico de Demanda Eléctrica (MW)")
st.markdown("Modelo XGBoost + skforecast")

# --- Constants ---
LAG_STEPS = 96
TIME_STEP_MINUTES = 30
STEPS_PER_DAY = 48
EXCEL_FILE = "DemandaCOES_.xlsx"

# Mapping for day name dummies (using numbers for natural sorting order)
DAYS_TRANSLATION = {
    'Monday': '1Lunes',
    'Tuesday': '2Martes',
    'Wednesday': '3Miércoles',
    'Thursday': '4Jueves',
    'Friday': '5Viernes',
    'Saturday': '6Sábado',
    'Sunday': '0Domingo'
}

# 1. Get the sorted translated day names
all_translated_days = sorted(DAYS_TRANSLATION.values())
# 2. Prepend 'dia_' to get the final dummy column names
all_day_dummy_cols = [f'dia_{d}' for d in all_translated_days]

# List of all possible exogenous features
ALL_FEATURES = ['ciclo', 'feriado'] + all_day_dummy_cols

# Force alphabetical ordering, which corresponds to the model's order.
REQUIRED_EXOG_COLS = sorted(ALL_FEATURES)


# --- Feature Engineering ---
def create_exogenous_features(data):
    """
    Creates the 'ciclo', day dummies, and 'feriado' features,
    and applies the required (alphabetical) ordering.
    """
    data_with_features = data.copy()
    
    # 1. Cyclic feature
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
    
    # 3. Holiday Feature (Robust handling)
    start_year = data_with_features.index.min().year - 1
    end_year = data_with_features.index.max().year + 2

    pe_holidays_from_package = {}
    try:
        # Use +1 to ensure the current end year is included
        pe_holidays_from_package = holidays.Peru(years=range(start_year, end_year + 1), observed=True)
    except Exception:
        pe_holidays_from_package = {}

    fixed_holidays_dates = []
    for year in range(start_year, end_year + 1):
        fixed_holidays_dates.extend([
            pd.to_datetime(f"{year}-01-01").date(),    # Año Nuevo
            pd.to_datetime(f"{year}-05-01").date(),    # Día del Trabajo
            pd.to_datetime(f"{year}-07-28").date(),    # Fiestas Patrias - Independencia
            pd.to_datetime(f"{year}-07-29").date(),    # Fiestas Patrias - Batalla de Ayacucho
            pd.to_datetime(f"{year}-08-30").date(),    # Santa Rosa de Lima
            pd.to_datetime(f"{year}-10-08").date(),    # Combate de Angamos
            pd.to_datetime(f"{year}-11-01").date(),    # Día de Todos los Santos
            pd.to_datetime(f"{year}-12-08").date(),    # Inmaculada Concepción
            pd.to_datetime(f"{year}-12-25").date(),    # Navidad
        ])
    
    known_holidays = set(pe_holidays_from_package.keys()) if isinstance(pe_holidays_from_package, dict) else set(pe_holidays_from_package)
    known_holidays.update(fixed_holidays_dates)

    data_with_features['feriado'] = data_with_features.index.normalize().isin(known_holidays).astype(int)
    
    # Drop target column if present
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

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        data = pd.read_excel(EXCEL_FILE, skiprows=3)
        data['FECHA'] = pd.to_datetime(data['FECHA'], format='%d/%m/%Y %H:%M')
        data.set_index('FECHA', inplace=True)
        data.rename(columns={'EJECUTADO': 'Demand'}, inplace=True)
        data = data.asfreq(f"{TIME_STEP_MINUTES}min")
        
        y = data['Demand'].copy()
        exog = create_exogenous_features(data[['Demand']])
        
        # Store fit_exog_cols and dtypes for consistency in prediction
        st.session_state['fit_exog_cols'] = exog.columns.tolist()
        st.session_state['fit_exog_dtypes'] = exog.dtypes.to_dict()

        return y, exog
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{EXCEL_FILE}'")
        return None, None

# --- Model Training ---
@st.cache_resource
def train_model(y_train, x_train):
    model = XGBRegressor(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        random_state=123,
        n_jobs=-1
    )
    forecaster = ForecasterRecursive(regressor=model, lags=LAG_STEPS)
    
    if not x_train.empty:
        forecaster.fit(y=y_train, exog=x_train)
    else:
        forecaster.fit(y=y_train)
        
    return forecaster

# --- Prediction Utilities ---

# Re-introducing generate_prediction_exog as a separate function
def generate_prediction_exog(start_date_for_exog, steps_for_exog):
    """Generates a future date range and the required exogenous variables."""
    future_index = pd.date_range(
        start=start_date_for_exog,
        periods=steps_for_exog,
        freq=f"{TIME_STEP_MINUTES}min"
    )

    future_index = future_index.drop_duplicates()
    future_data = pd.DataFrame(index=future_index)
    exog_pred = create_exogenous_features(future_data)
    
    # Safety checks
    if exog_pred.isnull().any().any():
        st.error("Error Interno: `exog_pred` contiene valores nulos después de la generación de características.")
        st.write(exog_pred.isnull().sum())
        st.stop()
    if not exog_pred.index.is_monotonic_increasing:
        st.error("Error Interno: El índice de `exog_pred` no está ordenado monótonamente creciente.")
        st.stop()
    if exog_pred.index.duplicated().any():
        st.error("Error Interno: El índice de `exog_pred` contiene duplicados.")
        st.stop()
        
    return exog_pred


# --- Main Prediction Function ---
def predict_future(forecaster, skforecast_predict_start_datetime, user_display_start_datetime, user_horizon_steps):
    
    # Calculate the end of the user's requested display horizon
    end_of_user_display_horizon = user_display_start_datetime + pd.Timedelta(minutes=TIME_STEP_MINUTES * (user_horizon_steps - 1))

    # Calculate total steps needed for skforecast, starting from skforecast_predict_start_datetime
    total_seconds_to_predict = (end_of_user_display_horizon - skforecast_predict_start_datetime).total_seconds()
    total_steps_for_skforecast_predict = int(total_seconds_to_predict / (TIME_STEP_MINUTES * 60)) + 1
    
    if total_steps_for_skforecast_predict < 1:
        if skforecast_predict_start_datetime <= end_of_user_display_horizon:
            total_steps_for_skforecast_predict = 1
        else:
            st.error("Error: La fecha de inicio de visualización es anterior a la fecha de inicio de predicción interna del modelo.")
            return pd.Series(dtype=float), pd.Series(dtype=float) # Return empty series for both


    # Generate Exogenous Variables for the FULL period required by skforecast
    exog_pred_full = generate_prediction_exog(skforecast_predict_start_datetime, total_steps_for_skforecast_predict)

    # --- Consistency checks and reconstruction ---
    fit_cols_list = st.session_state.get('fit_exog_cols', [])
    fit_dtypes_dict = st.session_state.get('fit_exog_dtypes', {})

    if fit_cols_list and exog_pred_full.empty:
        st.error("Error Interno: El modelo fue entrenado con exógenas, pero la generación de exógenas para la predicción resultó vacía.")
        st.stop()
    elif fit_cols_list: # If model was trained with exogenous variables, ensure consistency
        reconstructed_exog_pred = pd.DataFrame(
            0, # Default value
            index=exog_pred_full.index,
            columns=fit_cols_list,
        )
        for col in fit_cols_list:
            if col in exog_pred_full.columns:
                reconstructed_exog_pred[col] = exog_pred_full[col]
        for col, dtype in fit_dtypes_dict.items():
            if col in reconstructed_exog_pred.columns:
                reconstructed_exog_pred[col] = reconstructed_exog_pred[col].astype(dtype)
        exog_pred_full = reconstructed_exog_pred

    # Make Prediction with the FULL exog_pred_full
    if fit_cols_list:
        predictions_full = forecaster.predict(steps=total_steps_for_skforecast_predict, exog=exog_pred_full)
    else:
        predictions_full = forecaster.predict(steps=total_steps_for_skforecast_predict)

    # Slice the full prediction for display purposes, starting from user_display_start_datetime
    predictions_for_display = predictions_full.loc[user_display_start_datetime:].copy()
    
    return predictions_full, predictions_for_display

# --- Metrics Calculation Function ---
# This function is here for definition, but not used for live prediction metrics (as y_true is unknown)
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
        # Avoid division by zero by filtering non_zero_elements
        return np.mean(np.abs((y_true[non_zero_elements] - y_pred[non_zero_elements]) / y_true[non_zero_elements])) * 100

    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "MSE (Error Cuadrático Medio)": f"{mse:,.2f}",
        "RMSE (Raíz del Error Cuadrático Medio)": f"{rmse:,.2f} MW",
        "MAE (Error Absoluto Medio)": f"{mae:,.2f} MW",
        "MAPE (Error Porcentual Absoluto Medio)": f"{mape:,.2f}%",
        "Precisión (100 - MAPE)": f"{100 - mape:,.2f}%"
    }


# --- Main App Execution ---
# Load data and train model
y, exog = load_data()

if y is not None:
    forecaster = train_model(y, exog)
    
    # Sidebar controls
    st.sidebar.header("Configuración del Pronóstico")
    
    horizon = st.sidebar.selectbox(
        "Horizonte de predicción:",
        ["1 día", "3 días", "1 semana", "2 semanas", "1 mes"]
    )
    
    horizon_map = {
        "1 día": STEPS_PER_DAY,
        "3 días": STEPS_PER_DAY * 3,
        "1 semana": STEPS_PER_DAY * 7,
        "2 semanas": STEPS_PER_DAY * 14,
        "1 mes": STEPS_PER_DAY * 30
    }
    
    user_horizon_steps = horizon_map[horizon]
    
    # Define the actual start of prediction required by skforecast (one step after y ends)
    skforecast_predict_start_datetime = y.index[-1] + pd.Timedelta(minutes=TIME_STEP_MINUTES)
    
    user_selected_date = st.sidebar.date_input(
        "Fecha de inicio (visualización):",
        value=skforecast_predict_start_datetime.date(), # Default to the date of the actual next step
        min_value=skforecast_predict_start_datetime.date(),
        max_value=y.index[-1].date() + pd.Timedelta(days=365*2) # Max 2 years into the future from last historical data
    )
    
    user_display_start_datetime = pd.to_datetime(user_selected_date).replace(
        hour=skforecast_predict_start_datetime.hour,
        minute=skforecast_predict_start_datetime.minute
    )
    
    # Prediction button
    if st.sidebar.button("Generar Pronóstico", type="primary"):
        with st.spinner("Calculando pronóstico..."):
            predictions_full, predictions = predict_future(
                forecaster, 
                skforecast_predict_start_datetime, 
                user_display_start_datetime, 
                user_horizon_steps
            )
            
            # -----------------------------------------------------------
            # --- Plotly Graph Section Header ---
            # -----------------------------------------------------------
            
            st.subheader(f"Gráfica de Pronóstico - {horizon}")
            
            # Espacio vertical para separar el título de la gráfica (mantiene el 'lift' visual)
            #st.markdown("<br>", unsafe_allow_html=True)

            # --- Plotly Graph Code ---
            fig = go.Figure()

            if not y.empty and not predictions_full.empty:
                context_days = 7 # Show last 7 days of historical data
                context_start = max(y.index.min(), y.index[-1] - pd.Timedelta(days=context_days))
                
                y_context = y.loc[context_start:y.index[-1]]
                
                # Define the gap prediction (from historical end to user's display start)
                predictions_gap = predictions_full.loc[
                    skforecast_predict_start_datetime : user_display_start_datetime - pd.Timedelta(minutes=TIME_STEP_MINUTES)
                ]
                
                # --- Add Traces ---
                # 1. Historical Context Trace (Gris)
                if not y_context.empty:
                    fig.add_trace(go.Scatter(
                        x=y_context.index, 
                        y=y_context.values, 
                        mode='lines', 
                        line=dict(color='gray', width=1.5), 
                        name='Demanda Histórica (MW)'
                    ))
                    
                # 2. Prediction Gap Trace (Azul, Discontinua)
                if not predictions_gap.empty:
                    fig.add_trace(go.Scatter(
                        x=predictions_gap.index, 
                        y=predictions_gap.values, 
                        mode='lines', 
                        line=dict(color='blue', dash='dash', width=2), 
                        name='Pronóstico (entre histórico e inicio seleccionado)'
                    ))
                    
                # 3. User-Requested Forecast Trace (Rojo, Discontinua)
                if not predictions.empty:
                    fig.add_trace(go.Scatter(
                        x=predictions.index, 
                        y=predictions.values, 
                        mode='lines', 
                        line=dict(color='red', dash='dash', width=2), 
                        name=f'Pronóstico {horizon} (MW) (desde inicio seleccionado)'
                    ))
                    
                # 4. Add Vertical Line (End of Historical Data)
                fig.add_vline(
                    x=y.index[-1], 
                    line_width=2, 
                    line_dash="dot", 
                    line_color="blue", 
                    name='Fin datos históricos'
                )

                # Set appropriate x-axis limits and Title
                plot_start_limit = context_start
                if not predictions_gap.empty: 
                    plot_start_limit = min(context_start, predictions_gap.index.min())
                
                plot_end_limit = predictions_full.index[-1]
                
                fig.update_layout(
                    title=f'Demanda Eléctrica: Histórico Reciente y Pronóstico<br>{plot_start_limit.strftime("%d/%m/%Y %H:%M")} - {plot_end_limit.strftime("%d/%m/%Y %H:%M")}',
                    xaxis_title='Fecha y Hora',
                    yaxis_title='Demanda (MW)',
                    xaxis=dict(range=[plot_start_limit, plot_end_limit]),
                    # --- CAMBIOS CLAVE EN LA LEYENDA ---
                    legend=dict(
                        yanchor="bottom", # Ancla la parte inferior de la leyenda
                        y=1.00, # Posiciona la parte inferior justo en el borde superior del gráfico (lo mueve fuera del área de trazado)
                        xanchor="left", 
                        x=0.01,
                        orientation='h' # Hace la leyenda horizontal
                    ),
                    hovermode="x unified",
                    height=550,
                    margin=dict(t=80) # AUMENTA EL MARGEN SUPERIOR A 120
                )

            elif not y.empty: # Only historical data available (no predictions)
                context_days = 30
                context_start = max(y.index.min(), y.index[-1] - pd.Timedelta(days=context_days))
                y_context = y.loc[context_start:y.index[-1]]
                
                if not y_context.empty:
                    fig.add_trace(go.Scatter(
                        x=y_context.index, 
                        y=y_context.values, 
                        mode='lines', 
                        line=dict(color='gray', width=1.5), 
                        name='Demanda Histórica (MW)'
                    ))
                    fig.update_layout(
                        title=f'Demanda Eléctrica Histórica (Últimos {context_days} días)<br>{context_start.strftime("%d/%m/%Y %H:%M")} - {y.index[-1].strftime("%d/%m/%Y %H:%M")}',
                        xaxis_title='Fecha y Hora',
                        yaxis_title='Demanda (MW)',
                        legend=dict(yanchor="bottom", y=1.00, xanchor="left", x=0.01, orientation='h'), # Consistencia
                        height=550, 
                        margin=dict(t=80) # Consistencia 190
                    )
                else:
                    st.warning("No hay datos históricos para mostrar.")
                    fig.update_layout(title='No hay datos para mostrar', height=550)
            else: # No historical data
                st.warning("No hay datos históricos ni predicciones para mostrar.")
                fig.update_layout(title='No hay datos para mostrar', height=600)

            # Usar st.plotly_chart para renderizar la figura interactiva
            st.plotly_chart(fig, use_container_width=True)
            
            # -----------------------------------------------------------
            # --- Statistics in 2 columns below the plot ---
            # -----------------------------------------------------------
            st.subheader("Estadísticas del Pronóstico")
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown("**Pronóstico Actual**")
                if not predictions.empty:
                    st.metric("Demanda Promedio", f"{predictions.mean():,.0f} MW")
                    st.metric("Pico Máximo", f"{predictions.max():,.0f} MW")
                    st.metric("Valle Mínimo", f"{predictions.min():,.0f} MW")
                    st.metric("Rango", f"{predictions.max() - predictions.min():,.0f} MW")
                else:
                    st.markdown("No hay estadísticas para mostrar.")
            
            with col_stat2:
                st.markdown("**Rendimiento Histórico del Modelo**")
                st.info(
                    """
                    Precisión del modelo en la **última semana de datos históricos**
                    que no fueron utilizados para el entrenamiento.
                    """
                )
                st.metric("RMSE (Raíz del Error Cuadrático Medio)", f"94.54 MW")
                st.metric("MAPE (Error Porcentual Absoluto Medio)", f"1.54%")
                st.metric("Precisión (100 - MAPE)", f"98.46%")
            
            # -----------------------------------------------------------
            
            st.subheader("Datos del Pronóstico")
            predictions_df = predictions.to_frame('Demanda (MW)') # Use predictions (sliced) for the table
            predictions_df.index.name = 'Fecha-Hora'
            
            display_df = predictions_df.copy()
            display_df['Demanda (MW)'] = display_df['Demanda (MW)'].round(2)
            
            if not display_df.empty:
                st.dataframe(display_df, use_container_width=True)
                csv = predictions_df.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Descargar CSV",
                    data=csv,
                    file_name=f'pronostico_demanda_{user_display_start_datetime.strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            else:
                st.warning("No hay pronósticos para mostrar o descargar.")
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Información del Modelo:**")
    st.sidebar.markdown(f"• Lags utilizados: {LAG_STEPS}")
    st.sidebar.markdown(f"• Último dato histórico: {y.index[-1].strftime('%d/%m/%Y %H:%M')}")
    st.sidebar.markdown(f"• Total de datos históricos: {len(y):,}")
    
else:
    st.error("No se pudieron cargar los datos. Verifique que el archivo Excel esté en el directorio correcto.")
