import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Semilla maestra para reproducibilidad total
SEED = 42
SAMPLE_SIZE = 25000  # Cumple con utilizar al menos 10,000 registros

def main():
    ruta_csv = "data/creditcard.csv"
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_csv}")

    print("="*65)
    print("1. EXPLORACIÓN INICIAL DEL DATASET (Credit Card Fraud Detection)")
    print("="*65)
    df = pd.read_csv(ruta_csv)

    n_filas, n_cols = df.shape
    nulos_totales = df.isnull().sum().sum()
    duplicados = df.duplicated().sum()

    print(f"- Registros originales       : {n_filas:,}")
    print(f"- Número de columnas totales : {n_cols} ({n_cols - 1} predictoras + 1 objetivo 'Class')")
    print(f"- Valores faltantes (NaN)    : {nulos_totales}")
    print(f"- Registros duplicados       : {duplicados:,}")

    # Distribución original
    conteo_orig = df['Class'].value_counts()
    prop_orig = df['Class'].value_counts(normalize=True) * 100
    print("\nDistribución original de clases:")
    print(f"  * Clase 0 (Legítima)   : {conteo_orig[0]:,} ({prop_orig[0]:.4f}%)")
    print(f"  * Clase 1 (Fraude)     : {conteo_orig[1]:,} ({prop_orig[1]:.4f}%)")

    # Justificación técnica de limpieza: eliminar duplicados exactos
    # Evita que una misma transacción repetida caiga simultáneamente en Train y Test (Data Leakage)
    print("\n" + "="*65)
    print("2. LIMPIEZA DE DATOS")
    print("="*65)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    conteo_clean = df_clean['Class'].value_counts()
    prop_clean = df_clean['Class'].value_counts(normalize=True) * 100
    print(f"Se eliminaron {duplicados:,} duplicados exactos para evitar sesgo y fuga de datos entre particiones.")
    print(f"- Registros tras limpieza    : {len(df_clean):,}")
    print(f"  * Clase 0 (Legítima)   : {conteo_clean[0]:,} ({prop_clean[0]:.4f}%)")
    print(f"  * Clase 1 (Fraude)     : {conteo_clean[1]:,} ({prop_clean[1]:.4f}%)")

    # Muestreo estratificado para viabilidad computacional (mínimo exigido: 10,000)
    print("\n" + "="*65)
    print(f"3. MUESTREO ESTRATIFICADO ({SAMPLE_SIZE:,} REGISTROS)")
    print("="*65)
    df_sample, _ = train_test_split(
        df_clean,
        train_size=SAMPLE_SIZE,
        stratify=df_clean['Class'],
        random_state=SEED
    )
    df_sample = df_sample.reset_index(drop=True)
    conteo_sample = df_sample['Class'].value_counts()
    prop_sample = df_sample['Class'].value_counts(normalize=True) * 100
    print(f"- Muestra de trabajo seleccionada : {len(df_sample):,} registros")
    print(f"  * Clase 0 (Legítima)   : {conteo_sample[0]:,} ({prop_sample[0]:.4f}%)")
    print(f"  * Clase 1 (Fraude)     : {conteo_sample[1]:,} ({prop_sample[1]:.4f}%)")

    # Separar variables predictoras (X) de la variable objetivo (y)
    X = df_sample.drop(columns=['Class'])
    y = df_sample['Class']
    assert 'Class' not in X.columns, "ERROR: La columna Class no debe estar en X"

    # División 70% Entrenamiento y 30% Prueba (Estratificada y con semilla fija)
    print("\n" + "="*65)
    print("4. PARTICIÓN ENTRENAMIENTO (70%) Y PRUEBA (30%)")
    print("="*65)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=SEED
    )

    print(f"- Conjunto de Entrenamiento (70%) : {len(X_train):,} registros")
    print(f"  * Legítimas (0) : {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.4f}%)")
    print(f"  * Fraudes (1)   : {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.4f}%)")

    print(f"\n- Conjunto de Prueba Reservado (30%): {len(X_test):,} registros")
    print(f"  * Legítimas (0) : {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.4f}%)")
    print(f"  * Fraudes (1)   : {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.4f}%)")

    # Guardar las particiones para garantizar que TODOS los experimentos usen exactamente los mismos datos
    os.makedirs("data/procesado", exist_ok=True)
    X_train.to_parquet("data/procesado/X_train.parquet") if False else X_train.to_pickle("data/procesado/X_train.pkl")
    X_test.to_pickle("data/procesado/X_test.pkl")
    y_train.to_pickle("data/procesado/y_train.pkl")
    y_test.to_pickle("data/procesado/y_test.pkl")
    print("\n[OK] Particiones guardadas exitosamente en 'data/procesado/'.")

if __name__ == "__main__":
    main()
