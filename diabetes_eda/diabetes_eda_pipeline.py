"""
Análisis Exploratorio de Datos y Pipeline de ML
Dataset: Diabetes Dataset
Autor: [Tu nombre]
Fecha: 2026-02-03
"""

# ============================================================================
# CELDA 1: Importación de librerías
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Librerías importadas correctamente")


# ============================================================================
# CELDA 2: Carga y descripción del conjunto de datos
# ============================================================================

"""
## Descripción del Conjunto de Datos

**Fuente:** Scikit-learn Diabetes Dataset
**Enlace:** https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

El conjunto de datos de Diabetes fue obtenido originalmente del estudio:
Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004)
"Least Angle Regression," Annals of Statistics (with discussion), 407-499.

Este dataset contiene información de 442 pacientes diabéticos y incluye:
- 10 variables predictoras (edad, sexo, índice de masa corporal, presión arterial
  y seis mediciones de suero sanguíneo)
- 1 variable objetivo que es una medida cuantitativa de la progresión de la
  enfermedad un año después del baseline
"""

# Cargar el dataset
diabetes = load_diabetes()

# Crear un DataFrame con los datos
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print("Dataset cargado exitosamente")
print(f"Dimensiones del dataset: {df.shape}")
print(f"Número de muestras: {df.shape[0]}")
print(f"Número de características: {df.shape[1] - 1}")


# ============================================================================
# CELDA 3: Descripción del problema a resolver
# ============================================================================

"""
## Problema a Resolver

**Tipo de problema:** Regresión

**Objetivo:** Predecir la progresión de la diabetes en pacientes un año después
de la medición inicial (baseline), utilizando características demográficas y
mediciones clínicas.

**Variable objetivo (target):** Medida cuantitativa de la progresión de la
enfermedad (valores entre 25 y 346)

**Variables predictoras:**
- age: Edad
- sex: Sexo
- bmi: Índice de masa corporal (Body Mass Index)
- bp: Presión arterial promedio (Average Blood Pressure)
- s1: tc, Colesterol total sérico
- s2: ldl, Lipoproteínas de baja densidad
- s3: hdl, Lipoproteínas de alta densidad
- s4: tch, Relación colesterol total/HDL
- s5: ltg, Logaritmo de los niveles de triglicéridos séricos
- s6: glu, Nivel de glucosa en sangre

**Nota:** Todas las variables han sido centradas y escaladas previamente por
scikit-learn.
"""

print("Problema definido: Regresión para predecir progresión de diabetes")


# ============================================================================
# CELDA 4: EDA - Información general del dataset
# ============================================================================

"""
## Análisis Exploratorio de Datos (EDA)

### 1. Información General del Dataset
"""

print("=" * 80)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 80)

# Información del DataFrame
print("\n1. Información del DataFrame:")
print(df.info())

print("\n2. Primeras 5 filas del dataset:")
print(df.head())

print("\n3. Estadísticas descriptivas:")
print(df.describe())

print("\n4. Tipos de datos:")
print(df.dtypes)


# ============================================================================
# CELDA 5: EDA - Identificación de columnas y selección de variables
# ============================================================================

"""
### 2. Identificación y Selección de Columnas
"""

print("=" * 80)
print("IDENTIFICACIÓN Y SELECCIÓN DE COLUMNAS")
print("=" * 80)

# Todas las columnas
all_columns = df.columns.tolist()
print(f"\nColumnas totales: {all_columns}")

# Separar features y target
feature_columns = [col for col in all_columns if col != 'target']
target_column = 'target'

print(f"\nColumnas de características (features): {feature_columns}")
print(f"Columna objetivo (target): {target_column}")

print(f"\n**Decisión:** Utilizaremos todas las {len(feature_columns)} características")
print("disponibles ya que todas son mediciones clínicas relevantes para predecir")
print("la progresión de la diabetes.")


# ============================================================================
# CELDA 6: EDA - Distribución de cada variable
# ============================================================================

"""
### 3. Visualización de la Distribución de Variables
"""

print("=" * 80)
print("DISTRIBUCIÓN DE VARIABLES")
print("=" * 80)

# Crear figura con subplots para todas las variables
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Distribución de Variables del Dataset Diabetes', fontsize=16, y=1.00)

# Aplanar el array de axes para facilitar iteración
axes = axes.ravel()

# Graficar histogramas para cada columna
for idx, col in enumerate(df.columns):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Valor')
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].grid(True, alpha=0.3)

# Ocultar el último subplot (ya que tenemos 11 variables y 12 posiciones)
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('distribucion_variables.png', dpi=300, bbox_inches='tight')
print("\nGráfica guardada como 'distribucion_variables.png'")
plt.show()


# ============================================================================
# CELDA 7: EDA - Distribución de la variable objetivo
# ============================================================================

"""
### 4. Análisis Detallado de la Variable Objetivo
"""

print("=" * 80)
print("ANÁLISIS DE LA VARIABLE OBJETIVO (TARGET)")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribución de la Variable Objetivo: Progresión de Diabetes',
             fontsize=14, fontweight='bold')

# Histograma
axes[0].hist(df['target'], bins=30, edgecolor='black', alpha=0.7, color='salmon')
axes[0].set_title('Histograma')
axes[0].set_xlabel('Progresión de la Diabetes')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df['target'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Media: {df["target"].mean():.2f}')
axes[0].axvline(df['target'].median(), color='green', linestyle='--',
                linewidth=2, label=f'Mediana: {df["target"].median():.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Boxplot
axes[1].boxplot(df['target'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[1].set_title('Boxplot')
axes[1].set_ylabel('Progresión de la Diabetes')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribucion_target.png', dpi=300, bbox_inches='tight')
print("\nGráfica guardada como 'distribucion_target.png'")
plt.show()

print(f"\nEstadísticas de la variable objetivo:")
print(f"  - Media: {df['target'].mean():.2f}")
print(f"  - Mediana: {df['target'].median():.2f}")
print(f"  - Desviación estándar: {df['target'].std():.2f}")
print(f"  - Mínimo: {df['target'].min():.2f}")
print(f"  - Máximo: {df['target'].max():.2f}")


# ============================================================================
# CELDA 8: EDA - Análisis de valores nulos
# ============================================================================

"""
### 5. Análisis de Valores Nulos o Faltantes
"""

print("=" * 80)
print("ANÁLISIS DE VALORES NULOS")
print("=" * 80)

# Contar valores nulos
null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 100

# Crear DataFrame con la información
null_info = pd.DataFrame({
    'Columna': df.columns,
    'Valores Nulos': null_counts.values,
    'Porcentaje (%)': null_percentages.values
})

print("\nTabla de valores nulos:")
print(null_info.to_string(index=False))

# Visualización de valores nulos
fig, ax = plt.subplots(figsize=(10, 6))
null_info_sorted = null_info.sort_values('Valores Nulos', ascending=True)
ax.barh(null_info_sorted['Columna'], null_info_sorted['Valores Nulos'],
        color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Número de Valores Nulos')
ax.set_ylabel('Columnas')
ax.set_title('Valores Nulos por Columna', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('valores_nulos.png', dpi=300, bbox_inches='tight')
print("\nGráfica guardada como 'valores_nulos.png'")
plt.show()

if df.isnull().sum().sum() == 0:
    print("\n✓ Resultado: El dataset NO contiene valores nulos.")
else:
    print(f"\n✗ El dataset contiene {df.isnull().sum().sum()} valores nulos en total.")


# ============================================================================
# CELDA 9: EDA - Matriz de correlación
# ============================================================================

"""
### 6. Matriz de Correlación
"""

print("=" * 80)
print("MATRIZ DE CORRELACIÓN")
print("=" * 80)

# Calcular matriz de correlación
correlation_matrix = df.corr()

# Visualizar matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - Dataset Diabetes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
print("\nGráfica guardada como 'matriz_correlacion.png'")
plt.show()

# Correlación con la variable objetivo
print("\nCorrelación de cada feature con el target (ordenado):")
target_corr = correlation_matrix['target'].drop('target').sort_values(ascending=False)
print(target_corr)


# ============================================================================
# CELDA 10: EDA - Hallazgos principales
# ============================================================================

"""
## Hallazgos Principales del EDA

### Resumen de Hallazgos:

1. **Estructura del Dataset:**
   - El dataset contiene 442 muestras y 10 características predictoras
   - No hay valores nulos o faltantes en ninguna columna
   - Todas las variables son numéricas de tipo float64
   - Las variables ya han sido normalizadas previamente por scikit-learn

2. **Variable Objetivo (target):**
   - Rango de valores: 25 a 346
   - Distribución aproximadamente normal con ligera asimetría positiva
   - Media: ~152.13, Mediana: ~140.50
   - Presenta algunos valores atípicos en el extremo superior

3. **Distribución de Features:**
   - Todas las características muestran distribuciones centradas alrededor de 0
   - Las variables están normalizadas (centradas y escaladas)
   - La mayoría de las características muestran distribuciones aproximadamente normales
   - No se observan valores extremos preocupantes

4. **Correlaciones Importantes:**
   - BMI (s5) muestra la correlación más fuerte con el target (~0.59)
   - s5 (logaritmo de triglicéridos) también tiene buena correlación (~0.57)
   - bp (presión arterial) muestra correlación moderada (~0.44)
   - s6 (glucosa) tiene correlación moderada (~0.38)
   - age y sex muestran correlaciones más débiles
   - Algunas features están correlacionadas entre sí (multicolinealidad leve)

5. **Calidad de los Datos:**
   - Dataset limpio, sin necesidad de imputación de valores faltantes
   - No se requiere eliminación de outliers extremos
   - Las variables ya están en la misma escala
"""

print("=" * 80)
print("HALLAZGOS PRINCIPALES")
print("=" * 80)
print("\nVer comentarios en el código para descripción detallada de hallazgos")
print("\nPrincipales correlaciones con target:")
print(target_corr.head())


# ============================================================================
# CELDA 11: Definición del pipeline de preprocesamiento
# ============================================================================

"""
## Pipeline de Preprocesamiento

### Descripción de Transformaciones:

**Decisión de Preprocesamiento:**

Aunque los datos del dataset de diabetes de scikit-learn ya vienen normalizados,
vamos a crear un pipeline de preprocesamiento para demostrar buenas prácticas
y para que el código sea reutilizable con datos crudos.

**Transformaciones aplicadas:**

1. **StandardScaler en todas las features:**
   - **Justificación:** Aunque los datos ya están escalados, incluimos este paso
     para que el pipeline sea completo y funcione con datos sin procesar.
   - **Efecto:** Centra los datos (media=0) y escala (desviación estándar=1)
   - **Por qué:** Los algoritmos de regresión lineal y muchos otros se benefician
     de tener features en la misma escala, especialmente cuando se usan
     regularizaciones.

2. **No se aplica imputación:**
   - **Justificación:** El dataset no contiene valores nulos
   - Si hubiera valores nulos, usaríamos SimpleImputer con estrategia 'mean'

3. **No se aplica codificación:**
   - **Justificación:** Todas las variables son numéricas
   - Si hubiera variables categóricas, usaríamos OneHotEncoder

**Pipeline final:** StandardScaler → Modelo de Regresión Lineal
"""

print("=" * 80)
print("DEFINICIÓN DEL PIPELINE DE PREPROCESAMIENTO")
print("=" * 80)

# Definir las transformaciones para las columnas numéricas
numeric_features = feature_columns

# Crear el transformador para características numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Crear el ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

print("\nPipeline de preprocesamiento creado:")
print(preprocessor)

print("\nTransformaciones aplicadas:")
print("  - StandardScaler: Normalización de todas las características numéricas")
print(f"  - Características procesadas: {len(numeric_features)}")


# ============================================================================
# CELDA 12: Partición del conjunto de datos
# ============================================================================

"""
## Partición del Conjunto de Datos

Se divide el dataset en conjuntos de entrenamiento (80%) y prueba (20%)
con una semilla aleatoria fija para reproducibilidad.
"""

print("=" * 80)
print("PARTICIÓN DEL CONJUNTO DE DATOS")
print("=" * 80)

# Separar features (X) y target (y)
X = df[feature_columns]
y = df[target_column]

print(f"\nForma de X (features): {X.shape}")
print(f"Forma de y (target): {y.shape}")

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nConjunto de entrenamiento:")
print(f"  - X_train: {X_train.shape}")
print(f"  - y_train: {y_train.shape}")

print(f"\nConjunto de prueba:")
print(f"  - X_test: {X_test.shape}")
print(f"  - y_test: {y_test.shape}")

print(f"\nPorcentaje de datos en entrenamiento: {len(X_train)/len(X)*100:.1f}%")
print(f"Porcentaje de datos en prueba: {len(X_test)/len(X)*100:.1f}%")


# ============================================================================
# CELDA 13: Creación y entrenamiento del modelo
# ============================================================================

"""
## Creación del Modelo de Machine Learning

Se crea un pipeline completo que integra el preprocesamiento y el modelo predictivo.
Esto evita el "data leakage" ya que las transformaciones solo se ajustan con los
datos de entrenamiento.

**Modelo utilizado:** Regresión Lineal (LinearRegression)
**Justificación:** Es apropiado para problemas de regresión y sirve como baseline.
"""

print("=" * 80)
print("CREACIÓN Y ENTRENAMIENTO DEL MODELO")
print("=" * 80)

# Crear el pipeline completo (preprocesamiento + modelo)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("\nPipeline completo creado:")
print(model_pipeline)

# Entrenar el modelo
print("\nEntrenando el modelo...")
model_pipeline.fit(X_train, y_train)
print("✓ Modelo entrenado exitosamente")

# Mostrar los coeficientes del modelo
coefficients = model_pipeline.named_steps['regressor'].coef_
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nCoeficientes del modelo (importancia de features):")
print(feature_importance.to_string(index=False))


# ============================================================================
# CELDA 14: Realizar predicciones
# ============================================================================

"""
## Realizar Predicciones con el Conjunto de Prueba

Se utilizan los datos de prueba para generar predicciones con el modelo entrenado.
"""

print("=" * 80)
print("PREDICCIONES CON EL CONJUNTO DE PRUEBA")
print("=" * 80)

# Realizar predicciones
y_pred = model_pipeline.predict(X_test)

print(f"\nNúmero de predicciones realizadas: {len(y_pred)}")

# Mostrar algunas predicciones vs valores reales
comparison_df = pd.DataFrame({
    'Valor Real': y_test.values[:10],
    'Predicción': y_pred[:10],
    'Diferencia': y_test.values[:10] - y_pred[:10]
})

print("\nPrimeras 10 predicciones vs valores reales:")
print(comparison_df.to_string(index=False))

# Visualizar predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=80)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Predicción perfecta')
plt.xlabel('Valores Reales', fontsize=12)
plt.ylabel('Predicciones', fontsize=12)
plt.title('Predicciones vs Valores Reales - Conjunto de Prueba',
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
print("\nGráfica guardada como 'predicciones_vs_reales.png'")
plt.show()


# ============================================================================
# CELDA 15: Resumen final
# ============================================================================

"""
## Resumen Final

Este análisis ha completado las siguientes etapas:

1. ✓ Carga y descripción del dataset de Diabetes
2. ✓ Definición del problema de regresión
3. ✓ Análisis exploratorio de datos (EDA):
   - Identificación de tipos de datos
   - Selección de características
   - Visualización de distribuciones
   - Análisis de valores nulos
   - Matriz de correlación
   - Descripción de hallazgos
4. ✓ Creación del pipeline de preprocesamiento con StandardScaler
5. ✓ Partición de datos en entrenamiento (80%) y prueba (20%)
6. ✓ Entrenamiento del modelo de Regresión Lineal
7. ✓ Generación de predicciones en el conjunto de prueba

El pipeline está completo y listo para su uso en producción, evitando data leakage
al integrar preprocesamiento y modelo en una sola estructura.

**Próximos pasos sugeridos (fuera del alcance de esta actividad):**
- Evaluar métricas de desempeño (RMSE, MAE, R²)
- Probar otros algoritmos (Ridge, Lasso, Random Forest)
- Realizar validación cruzada
- Optimizar hiperparámetros
"""

print("=" * 80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 80)

print("\n✓ Todas las etapas han sido completadas")
print("\nArchivos generados:")
print("  - distribucion_variables.png")
print("  - distribucion_target.png")
print("  - valores_nulos.png")
print("  - matriz_correlacion.png")
print("  - predicciones_vs_reales.png")

print("\n" + "=" * 80)
print("FIN DEL ANÁLISIS")
print("=" * 80)
