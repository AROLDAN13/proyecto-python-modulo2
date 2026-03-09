import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data):
        self.df = data

    def clasificar_variables(self):
        numericas = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categoricas = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numericas, categoricas

    def obtener_estadisticas(self):
        return self.df.describe()

    def graficar_histograma(self, columna):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df[columna], kde=True, ax=ax, color='#1f77b4')
        ax.set_title(f'Distribución de {columna}') 
        return fig

    def graficar_barras(self, columna):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=self.df, x=columna, hue=columna, palette='viridis', legend=False)
        ax.set_title(f'Conteo de {columna}')
        plt.xticks(rotation=45)
        return fig
    
    def graficar_bivariado_num_cat(self, num_col, cat_col):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, hue=cat_col, palette='Set2', legend=False)
        ax.set_title(f'Análisis: {num_col} vs {cat_col}')
        return fig

st.set_page_config(page_title="Telco Churn EDA", layout="wide")

col1, col2, col3 = st.sidebar.columns([1, 2, 1])

with col2: 
    st.image("logo.png", width=150)

st.sidebar.title("Menú Principal")

opcion_menu = st.sidebar.radio(
    "Módulos:", 
    ["1. Home", "2. Carga del Dataset", "3. Análisis Exploratorio"]
)

# MÓDULO 1: HOME

if opcion_menu == "1. Home":
    st.title("Proyecto: Análisis Exploratorio de Churn")
    
    st.markdown("### Objetivo")
    st.write("Aplicar los conocimientos de Python para analizar, limpiar, transformar y visualizar los datos del dataset 'Telco Customer Churn', identificando patrones de fuga de clientes mediante el análisis exploratorio de los datos.")
    
    st.markdown("### Datos del Autor")
    st.write("- **Nombre completo:** Alex Enrique Roldan Talledo")
    st.write("- **Especialización:** Python for Analytics")
    st.write("- **Año:** 2026")
    
    st.markdown("### Tecnologías")
    st.write("Python, Pandas, Matplotlib, Seaborn y Streamlit.")

# MÓDULO 2: CARGA DEL DATASET

elif opcion_menu == "2. Carga del Dataset":
    st.title("Módulo de Carga de Datos")
    archivo = st.file_uploader("Sube el archivo .csv", type=["csv"])
    
    if archivo is not None:
        # Carga y limpieza inicial
        df = pd.read_csv(archivo)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') 
        
        st.session_state['dataset'] = df 
        
        st.success("Archivo cargado exitosamente.")
        st.write("### Vista previa")
        st.dataframe(df.head())
        
        st.write("### Dimensiones") 
        st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    else:
        st.warning("Es necesario cargar el archivo para continuar.")

# MÓDULO 3: EDA

elif opcion_menu == "3. Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos (EDA)")
    
    if 'dataset' not in st.session_state:
        st.error("Por favor, carga el dataset en el Módulo 2 primero.")
    else:
        df = st.session_state['dataset']
        analyzer = DataAnalyzer(df) 
        
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
            "1. Información", "2. Clasificación", "3. Estadísticas", "4. Nulos", 
            "5. Dist. Numéricas", "6. Dist. Categóricas", "7. Bivariado (N vs C)", 
            "8. Bivariado (C vs C)", "9. Dinámico", "10. Hallazgos"
        ])
        
        with t1:
            st.header("Ítem 1: Información General")
            st.write("**Tipos de datos y Conteo de Nulos por campo:**")
            
            info_df = pd.DataFrame({
                'Tipo de dato': df.dtypes.astype(str),
                'Valores nulos': df.isnull().sum()
            })
            
            st.dataframe(info_df)
            
        with t2:
            st.header("Ítem 2: Clasificación de Variables")
            num, cat = analyzer.clasificar_variables()
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Numéricas ({len(num)}):**", num)
            with c2:
                st.write(f"**Categóricas ({len(cat)}):**", cat)
                
        with t3:
            st.header("Ítem 3: Estadísticas Descriptivas")
            st.write("**Resumen estadístico de las variables numéricas:**")
            
            estadisticas = df.describe()
            st.dataframe(estadisticas)
            
            st.markdown("""
            **Interpretación básica:**
            * **Media y Mediana:** La tarifa mensual promedio (`MonthlyCharges`) es de aproximadamente USD 64.76, mientras que la mediana (el percentil 50) es USD 70.35. Como la media es menor que la mediana, hay un sesgo hacia tarifas más altas.
            * **Dispersión:** La antigüedad (`tenure`) tiene una desviación estándar alta (24.5 meses), lo que indica que nuestra base está muy dispersa: tenemos clientes que acaban de entrar (mínimo 0 meses) y clientes muy fieles (máximo 72 meses).
            """)
            
        with t4:
            st.header("Ítem 4: Valores Faltantes")
            
            faltantes = df.isnull().sum()
            faltantes = faltantes[faltantes > 0]
            
            if not faltantes.empty:
                df_nulos = faltantes.to_frame(name="Cantidad de valores faltantes")
                
                st.dataframe(df_nulos)
                st.write("Se detectaron nulos en `TotalCharges` producto de la conversión de texto a número en clientes nuevos.")
            else:
                st.write("No hay valores faltantes.")
                
        with t5:
            st.header("Ítem 5: Distribución Numérica")
            num_cols, _ = analyzer.clasificar_variables()
            for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                if col in num_cols:
                    st.pyplot(analyzer.graficar_histograma(col))
                    
        with t6:
            st.header("Ítem 6: Distribución Categórica")
            st.pyplot(analyzer.graficar_barras('Contract'))
            st.pyplot(analyzer.graficar_barras('InternetService'))
            st.pyplot(analyzer.graficar_barras('PaymentMethod'))
                    
        with t7:
            st.header("Ítem 7: Numérico vs Categórico")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(analyzer.graficar_bivariado_num_cat('MonthlyCharges', 'Churn'))
            with c2:
                st.pyplot(analyzer.graficar_bivariado_num_cat('tenure', 'Churn'))
                
        with t8:
            st.header("Ítem 8: Categórico vs Categórico")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.countplot(data=df, x='Contract', hue='Churn', palette='Set1')
            ax.set_title("Relación entre Tipo de Contrato y Churn")
            st.pyplot(fig)
            
        with t9:
            st.header("Ítem 9: Análisis Dinámico")
            num_cols, cat_cols = analyzer.clasificar_variables()
            v_num = st.selectbox("Elige una variable numérica:", num_cols)
            v_cat = st.selectbox("Elige una variable categórica:", cat_cols)
            if v_num and v_cat:
                st.pyplot(analyzer.graficar_bivariado_num_cat(v_num, v_cat))
                
        with t10:
            st.header("Ítem 10: Hallazgos Clave")
            st.markdown("""
            1. **Fidelización temprana:** Los clientes con `tenure` bajo (primeros meses) concentran la mayor fuga.
            2. **Contratos vulnerables:** El contrato `Month-to-month` es el mayor factor de riesgo comparado con los anuales.
            3. **Sensibilidad al precio:** Cargos mensuales (`MonthlyCharges`) más altos se asocian con mayor probabilidad de abandono.
            4. **Servicios de fibra:** Los clientes con `Fiber optic` tienen una tasa de churn preocupantemente alta.
            5. **Retención por servicios extra:** Carecer de `TechSupport` o `OnlineSecurity` facilita la salida del cliente hacia la competencia.
            """)
            if st.checkbox("Ver Matriz de Correlación"):
                fig, ax = plt.subplots(figsize=(6,4))
                sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True, cmap='coolwarm')
                st.pyplot(fig)