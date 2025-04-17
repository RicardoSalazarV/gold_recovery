import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Análisis de Recuperación de Oro",
    page_icon="🏆",
    layout="wide"
)

# Función para cargar el dataset 
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Datos cargados desde el archivo subido por el usuario")
        else:
            # Lista de posibles rutas para encontrar el archivo
            possible_paths = [
                "datasets/gold_recovery_train.csv",  # En carpeta datasets
                "gold_recovery_train.csv",           # En la raíz del proyecto
                "./gold_recovery_train.csv",         # Ruta relativa explícita
                os.path.join("datasets", "gold_recovery_train.csv")  # Usando os.path para compatibilidad
            ]
            
            # Para desarrollo local 
            if os.path.exists("C:\\Users\\ricar\\Desktop\\gold_recovery\\datasets\\gold_recovery_train.csv"):
                possible_paths.insert(0, "C:\\Users\\ricar\\Desktop\\gold_recovery\\datasets\\gold_recovery_train.csv")
            
            
            df = None
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    st.success(f"Datos cargados correctamente desde: {path}")
                    break
                except FileNotFoundError:
                    continue
                
            if df is None:
                st.info("No se encontró el archivo de datos. Por favor, carga un archivo CSV.")
                return pd.DataFrame()
        
        # Preprocesamiento
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        
        if 'rougher.output.recovery' in df.columns:
            df['recovery_rate'] = df['rougher.output.recovery']
        
        return df

    except pd.errors.EmptyDataError:
        st.error("El archivo está vacío.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

# Función para mostrar estadísticas 
def show_statistics(filtered_df):
    with st.expander("📊 Estadísticas generales", expanded=True):
        st.write(f"### Datos Filtrados ({len(filtered_df)} registros)")
        st.dataframe(filtered_df.describe())
        show_data = st.checkbox("Mostrar datos completos")
        if show_data:
            st.dataframe(filtered_df)

# Función para construir histograma
def build_histogram(filtered_df, column):
    st.write(f"### Histograma de {column}")
    fig_hist = px.histogram(
        filtered_df, 
        x=column, 
        title=f"Histograma de {column}",
        labels={column: column},
        nbins=30
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Función para construir un gráfico de dispersión
def build_scatter_plot(filtered_df):
    st.write("### Gráfico de dispersión")
    
    # Seleccionar columnas para el gráfico
    numeric_columns = filtered_df.select_dtypes(include=['float', 'int']).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Se necesitan al menos dos columnas numéricas para crear un gráfico de dispersión.")
        return
    
    x_column = st.selectbox("Selecciona columna X", options=numeric_columns, index=0)
    
    # Determinar índice para y_column 
    y_index = numeric_columns.index('recovery_rate') if 'recovery_rate' in numeric_columns else 0
    if y_index == 0 and len(numeric_columns) > 1:  # Evitar que x e y sean iguales
        y_index = 1
    
    y_column = st.selectbox("Selecciona columna Y", options=numeric_columns, index=y_index)
    
    # Opciones para color
    color_options = ['Ninguno'] + filtered_df.columns.tolist()
    color_column = st.selectbox("Columna para color (opcional)", options=color_options)
    color_param = None if color_column == 'Ninguno' else color_column
    
    # Crear el gráfico de dispersión
    fig_scatter = px.scatter(
        filtered_df, 
        x=x_column,
        y=y_column, 
        color=color_param,
        title=f"Relación entre {y_column} y {x_column}",
        labels={x_column: x_column, y_column: y_column}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Función para mostrar la distribución
def show_category_distribution(filtered_df, column):
    if column not in filtered_df.columns:
        st.warning(f"La columna '{column}' no existe en el dataset.")
        return
        
    st.write(f"### Distribución de {column}")
    
    # Verificar si hay demasiadas categorías
    if filtered_df[column].nunique() > 20:
        st.warning(f"La columna '{column}' tiene demasiadas categorías ({filtered_df[column].nunique()}). Mostrando las 20 más frecuentes.")
        top_categories = filtered_df[column].value_counts().nlargest(20).index
        plot_df = filtered_df[filtered_df[column].isin(top_categories)].copy()
    else:
        plot_df = filtered_df
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=plot_df, y=column, order=plot_df[column].value_counts().index, palette='muted', ax=ax)
    ax.set_title(f"Distribución de {column}")
    ax.set_xlabel("Frecuencia")
    ax.set_ylabel(column)
    st.pyplot(fig)

# Función para mostrar un boxplot 
def show_boxplot_by_group(filtered_df, value_column, group_column):
    if value_column not in filtered_df.columns or group_column not in filtered_df.columns:
        st.warning(f"Una de las columnas seleccionadas no existe en el dataset.")
        return
        
    st.write(f"### {value_column} por {group_column}")
    
    # Verificar si hay demasiadas categorías
    if filtered_df[group_column].nunique() > 10:
        st.warning(f"La columna '{group_column}' tiene demasiadas categorías. Mostrando las 10 más frecuentes.")
        top_categories = filtered_df[group_column].value_counts().nlargest(10).index
        plot_df = filtered_df[filtered_df[group_column].isin(top_categories)].copy()
    else:
        plot_df = filtered_df
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, x=group_column, y=value_column, palette='coolwarm', ax=ax)
    ax.set_title(f"{value_column} por {group_column}")
    ax.set_xlabel(group_column)
    ax.set_ylabel(value_column)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Función para mostrar correlaciones entre variables numéricas
def show_correlation_heatmap(filtered_df):
    st.write("### Matriz de Correlación")
    
    # Seleccionar solo columnas numéricas
    numeric_df = filtered_df.select_dtypes(include=['float', 'int'])
    
    if numeric_df.shape[1] < 2:
        st.warning("Se necesitan al menos dos columnas numéricas para crear una matriz de correlación.")
        return
        
    # Limitar a un número manejable de columnas
    if numeric_df.shape[1] > 15:
        st.warning("Mostrando correlaciones para las 15 columnas más relevantes.")
        # Priorizar recovery_rate si existe
        if 'recovery_rate' in numeric_df.columns:
            # Calcular correlaciones con recovery_rate
            corr_with_target = numeric_df.corr()['recovery_rate'].abs().sort_values(ascending=False)
            selected_columns = corr_with_target.head(15).index.tolist()
            numeric_df = numeric_df[selected_columns]
        else:
            # Seleccionar aleatoriamente 15 columnas
            numeric_df = numeric_df.sample(n=min(15, numeric_df.shape[1]), axis=1)
    
    # Calcular la matriz de correlación
    corr_matrix = numeric_df.corr()
    
    # Crear el mapa de calor
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Correlación entre Variables Numéricas")
    st.pyplot(fig)

# Función principal de la aplicación
def main():
    # Título de la app
    st.title("🏆 Exploración Interactiva de Recuperación de Oro")
    
    # Cargar datos
    uploaded_file = st.file_uploader("Cargar archivo CSV de datos de recuperación de oro", type="csv")
    df = load_data(uploaded_file)
    
    # Validar si los datos fueron cargados 
    if df.empty:
        st.warning("No hay datos disponibles. Por favor, carga un archivo CSV válido.")
        return
    
    # Mostrar información básica del dataset
    st.write(f"### Información del Dataset")
    st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    
    # Sidebar con filtros interactivos
    st.sidebar.header("Filtros")
    
    # Filtro por rango de recuperación 
    if 'recovery_rate' in df.columns:
        recovery_min = float(df['recovery_rate'].min())
        recovery_max = float(df['recovery_rate'].max())
        recovery_range = st.sidebar.slider("Rango de recuperación", 
                                        min_value=recovery_min, 
                                        max_value=recovery_max, 
                                        value=(recovery_min, recovery_max))
    else:
        recovery_range = None
    
    # Detectar columnas categóricas
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Permitir seleccionar una columna categórica para filtrar
    category_filters = {}
    if categorical_columns:
        filter_column = st.sidebar.selectbox("Filtrar por categoría", 
                                          options=['Ninguno'] + categorical_columns)
        
        if filter_column != 'Ninguno':
            unique_values = df[filter_column].unique()
            selected_values = st.sidebar.multiselect(f"Valores de {filter_column}", 
                                                 options=unique_values, 
                                                 default=unique_values)
            if selected_values:
                category_filters[filter_column] = selected_values
    
    # Aplicar filtros
    filtered_df = df.copy()
    
    # Filtrar por rango de recuperación si existe
    if recovery_range and 'recovery_rate' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['recovery_rate'] >= recovery_range[0]) & 
            (filtered_df['recovery_rate'] <= recovery_range[1])
        ]
    
    # Aplicar filtros categóricos
    for column, values in category_filters.items():
        filtered_df = filtered_df[filtered_df[column].isin(values)]
    
    # Mostrar estadísticas generales
    show_statistics(filtered_df)
    
    # Sección de visualizaciones
    st.write("## Visualizaciones")
    
    # Pestañas para organizar las visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Relaciones", "Agrupaciones", "Correlaciones"])
    
    with tab1:
        # Histogramas de variables numéricas
        st.write("### Histogramas")
        numeric_columns = filtered_df.select_dtypes(include=['float', 'int']).columns.tolist()
        if numeric_columns:
            hist_column = st.selectbox("Selecciona columna para histograma", 
                                     options=numeric_columns,
                                     index=numeric_columns.index('recovery_rate') if 'recovery_rate' in numeric_columns else 0)
            build_histogram(filtered_df, hist_column)
        
        # Distribución de variables categóricas
        st.write("### Distribuciones Categóricas")
        if categorical_columns:
            cat_column = st.selectbox("Selecciona columna categórica", 
                                    options=categorical_columns)
            show_category_distribution(filtered_df, cat_column)
    
    with tab2:
        # Gráfico de dispersión
        build_scatter_plot(filtered_df)
    
    with tab3:
        # Boxplots por grupo
        st.write("### Análisis por Grupos")
        if numeric_columns and categorical_columns:
            value_col = st.selectbox("Selecciona variable numérica", 
                                   options=numeric_columns,
                                   index=numeric_columns.index('recovery_rate') if 'recovery_rate' in numeric_columns else 0)
            group_col = st.selectbox("Selecciona variable de agrupación", 
                                   options=categorical_columns)
            show_boxplot_by_group(filtered_df, value_col, group_col)
    
    with tab4:
        # Matriz de correlación
        show_correlation_heatmap(filtered_df)

    # Sección de conclusiones
    st.write("## Conclusiones")
    st.write("""
    Este análisis interactivo te permite explorar los datos de recuperación de oro y encontrar patrones 
    y relaciones entre las diferentes variables. Utiliza los filtros y visualizaciones para comprender 
    mejor los factores que afectan la tasa de recuperación.
    """)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()