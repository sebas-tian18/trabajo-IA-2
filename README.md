# Problema 2: Detección de Fraude usando Técnicas de Aprendizaje Supervisado
**Asignatura:** INFO1185 - Inteligencia Artificial  
**Profesor:** Dr. Ricardo Soto Catalán  

## 1. Requisitos y Versiones Principales
El proyecto fue desarrollado y validado en **Fedora Linux** utilizando un entorno virtual de Python.

- **Python:** 3.14.5
- **numpy:** 2.5.4
- **pandas:** 3.1.0
- **scikit-learn:** 1.9.1
- **imbalanced-learn:** 0.14.2
- **matplotlib:** 3.11.2
- **seaborn:** 0.13.2

Para instalar las dependencias exactas en un entorno virtual limpio:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## 2. Ubicación Esperada del Dataset
El archivo descargado de Kaggle (creditcard.csv) debe ubicarse en la ruta:
    Pregunta_2/data/creditcard.csv
(No se adjunta el dataset en la entrega según lo indicado en las instrucciones).

## 3. Semillas Utilizadas y Reproducibilidad
- **Semilla Maestra Global (SEED = 42):**
  - Muestreo estratificado de 25,000 registros (tras eliminar 1,081 duplicados exactos).
  - Partición estratificada 70% Entrenamiento (17,500) y 30% Prueba (7,500).
  - Validación cruzada StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
  - Compensación de desbalance SMOTE(sampling_strategy=0.20, random_state=42) y selector ExtraTreesClassifier(random_state=42).
- **Semillas por Árbol en CustomRandomForest:** A partir de random_state=42, un generador maestro deriva semillas distintas por árbol para asegurar diversidad en el muestreo bootstrap y en max_features, manteniendo reproducibilidad.

## 4. Orden de Ejecución
Ejecutar los scripts secuencialmente desde la raíz del proyecto:
    1. python3 01_exploracion_y_particion.py
    2. python3 custom_rf.py
    3. python3 02_entrenamiento_cv.py
    4. python3 03_evaluacion_final.py
