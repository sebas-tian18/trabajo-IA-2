# Problema 2: Detección de Fraude usando Técnicas de Aprendizaje Supervisado
**Asignatura:** INFO1185 - Inteligencia Artificial
**Profesor:** Dr. Ricardo Soto Catalán

## 1. Requisitos y versiones principales
El procedimiento completo fue ejecutado y verificado en Linux y en Windows, con resultados idénticos.

| Paquete          | Versión utilizada |
|------------------|-------------------|
| Python           | 3.13.3            |
| numpy            | 2.3.5             |
| pandas           | 3.0.3             |
| scikit-learn     | 1.9.0             |
| imbalanced-learn | 0.14.2            |
| matplotlib       | 3.10.7            |
| seaborn          | 0.13.2            |

Instalación de dependencias (`requirements.txt` contiene solo las librerías usadas por el proyecto):

    # Linux / macOS
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # Windows (PowerShell)
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## 2. Ubicación esperada del dataset
El archivo `creditcard.csv` (Credit Card Fraud Detection, Kaggle) debe ubicarse, relativo a la raíz del proyecto, en:

    data/creditcard.csv

El dataset no se adjunta en la entrega.

## 3. Orden de ejecución
Ejecutar los scripts en orden, desde la raíz del proyecto:

| Paso | Script                          | Qué hace                                                                                                   | Tiempo aprox. |
|------|---------------------------------|------------------------------------------------------------------------------------------------------------|---------------|
| 1    | `01_exploracion_y_particion.py` | Exploración, eliminación de duplicados, muestra estratificada de 25.000 registros y partición 70/30          | ~10 s         |
| 2    | `custom_rf.py` (opcional)       | Verificación rápida del Random Forest propio (bootstrap, semillas por árbol, % de registros únicos)          | ~5 s          |
| 3    | `02_entrenamiento_cv.py`        | Conteos de SMOTE por fold, Grid Search con CV estratificada de 5 folds para las 8 configuraciones y selección final | ~3-4 min      |
| 4    | `03_evaluacion_final.py`        | Evaluación en prueba, media ± desviación estándar por modelo (bootstrap), matrices de confusión y gráficos   | ~10 s         |
| 5    | `04_graficos_complementarios.py` | Gráficos para la presentación: exploración, importancia de variables y comparación árbol vs. Random Forest (solo CV) | ~30-40 s      |

Tiempos medidos en un equipo de 2 núcleos; varían según el computador.

## 4. Semillas y reproducibilidad
Todas las semillas derivan de `SEED = 42`:

- Muestreo estratificado de 25.000 registros (después de eliminar 1.081 duplicados exactos).
- Partición estratificada 70% entrenamiento (17.500) / 30% prueba (7.500).
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`: las mismas 5 particiones para los 4 clasificadores y las 2 condiciones.
- `SMOTE(sampling_strategy=0.20, k_neighbors=5, random_state=42)`, aplicado solo al entrenamiento de cada fold.
- Selector `ExtraTreesClassifier(n_estimators=50, random_state=42)` dentro del pipeline (top 15 características).
- `CustomRandomForest(random_state=42)`: un generador maestro deriva una semilla distinta para cada árbol (bootstrap y `max_features`).
- Bootstrap estratificado del conjunto de prueba: 1.000 réplicas con `random_state=42`.

## 5. Archivos generados en `resultados/`
| Archivo                                   | Contenido                                                                              |
|-------------------------------------------|----------------------------------------------------------------------------------------|
| `tabla_smote_por_fold.csv`                | Ejemplos por clase antes y después de SMOTE en cada fold (validación sin balancear)     |
| `tabla_validacion_cruzada.csv`            | Hiperparámetros elegidos, métricas de CV (media y std) y características seleccionadas  |
| `configuracion_seleccionada.csv`          | Configuración final elegida entre las 8, solo con resultados de CV                      |
| `tabla_cv_formateada.csv`                 | Resumen de CV en formato media ± std                                                    |
| `tabla_evaluacion_prueba.csv`             | Métricas puntuales en prueba y matriz de confusión (TP, FN, FP, TN) por configuración   |
| `tabla_prueba_media_std(_formateada).csv` | Media ± desviación estándar en prueba por modelo y condición (bootstrap estratificado)  |
| `matrices_confusion_test.png`             | Matrices de confusión de las 8 configuraciones en prueba                                |
| `grafico_comparativo_metricas.png`        | Sensibilidad y F1 en prueba con barras de error (bootstrap)                             |
| `exploracion_datos.png`                   | Distribución de clases y monto por clase (dataset completo, solo descriptivo)           |
| `importancia_variables.png`               | Importancia de las 30 variables según el selector; destaca las 15 seleccionadas         |
| `arbol_vs_rf.png`, `tabla_arbol_vs_rf*.csv` | Árbol vs. Random Forest propio fold por fold en CV (sensibilidad, precisión, falsas alarmas) |
| `curva_numero_arboles.png`, `tabla_numero_arboles.csv` | Efecto del número de árboles del Random Forest propio en CV                |
| `mejores_modelos_entrenados.pkl`          | Las 8 configuraciones reentrenadas en todo el entrenamiento (no se versiona en git)     |

## 6. Limitaciones conocidas
La muestra de 25.000 registros cumple el mínimo exigido (10.000), pero contiene 42 fraudes: 29 en entrenamiento (5-6 por fold de validación) y 13 en prueba. Por eso las desviaciones estándar de la sensibilidad son altas (±18-29% en CV y ±10-14% en prueba), y las diferencias entre modelos que dependen de 1 o 2 fraudes no deben interpretarse como concluyentes.