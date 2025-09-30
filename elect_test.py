import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursive

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

# --- Feature Engineering ---
def create_exogenous_features(data):
    data_with_features = data.copy()
    
    # Cyclic feature
    data_with_features['ciclo'] = data_with_features.index.map(
        lambda t: (t.hour * 60 + t.minute) / (24 * 60)
    )
    
    # Day of week dummies
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    spanish_days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    for i, (eng, esp) in enumerate(zip(day_names, spanish_days)):
        data_with_features[f'dia_{esp}'] = (data_with_features.index.day_name() == eng).astype(int)
    
    # Holiday feature
    start_year = data_with_features.index.min().year - 1
    end_year = data_with_features.index.max().year + 2
    pe_holidays = holidays.Peru(years=range(start_year, end_year), observed=True)
    data_with_features['feriado'] = data_with_features.index.normalize().isin(pe_holidays).astype(int)
    
    # Drop target column if present
    if 'Demand' in data_with_features.columns:
        exog = data_with_features.drop(columns=['Demand'])
    else:
        exog = data_with_features
        
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
        return data['Demand'], create_exogenous_features(data[['Demand']])
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
    forecaster.fit(y=y_train, exog=x_train)
    return forecaster

# --- Prediction ---
def predict_future(forecaster, start_date, steps):
    # Create future dates
    future_dates = pd.date_range(
        start=start_date,
        periods=steps,
        freq=f"{TIME_STEP_MINUTES}min"
    )
    
    # Create exogenous features for future
    future_data = pd.DataFrame(index=future_dates)
    exog_future = create_exogenous_features(future_data)
    
    # Make prediction
    return forecaster.predict(steps=steps, exog=exog_future)

# --- Main App ---
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
    
    steps = horizon_map[horizon]
    
    # Default start is next time step after last data point
    default_start = y.index[-1] + pd.Timedelta(minutes=TIME_STEP_MINUTES)
    
    start_date = st.sidebar.date_input(
        "Fecha de inicio:",
        value=default_start,
        min_value=default_start
    )
    
    # Add time component
    start_datetime = pd.to_datetime(start_date).replace(
        hour=default_start.hour,
        minute=default_start.minute
    )
    
    # Prediction button
    if st.sidebar.button("Generar Pronóstico", type="primary"):
        with st.spinner("Calculando pronóstico..."):
            predictions = predict_future(forecaster, start_datetime, steps)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Pronóstico - {horizon}")
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                predictions.plot(ax=ax, color='red', linewidth=2)
                ax.set_title(f'Pronóstico de Demanda Eléctrica\n{start_datetime.strftime("%d/%m/%Y")} - {predictions.index[-1].strftime("%d/%m/%Y")}')
                ax.set_xlabel('Fecha y Hora')
                ax.set_ylabel('Demanda (MW)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Estadísticas")
                st.metric("Demanda Promedio", f"{predictions.mean():.0f} MW")
                st.metric("Pico Máximo", f"{predictions.max():.0f} MW")
                st.metric("Valle Mínimo", f"{predictions.min():.0f} MW")
                st.metric("Rango", f"{predictions.max() - predictions.min():.0f} MW")
            
            # Data table
            st.subheader("Datos del Pronóstico")
            predictions_df = predictions.to_frame('Demanda (MW)')
            predictions_df.index.name = 'Fecha-Hora'
            
            # Format for display
            display_df = predictions_df.copy()
            display_df['Demanda (MW)'] = display_df['Demanda (MW)'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = predictions_df.to_csv().encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                data=csv,
                file_name=f'pronostico_demanda_{start_datetime.strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Información del Modelo:**")
    st.sidebar.markdown(f"• Lags utilizados: {LAG_STEPS}")
    st.sidebar.markdown(f"• Último dato: {y.index[-1].strftime('%d/%m/%Y %H:%M')}")
    st.sidebar.markdown(f"• Total de datos: {len(y):,}")
    
else:
    st.error("No se pudieron cargar los datos. Verifique que el archivo Excel esté en el directorio correcto.")
