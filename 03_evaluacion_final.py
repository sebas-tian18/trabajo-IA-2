import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)

SEED = 42


def calcular_metricas(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensibilidad = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    especificidad = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    exactitud = accuracy_score(y_true, y_pred)
    fnr = 1.0 - sensibilidad  # FNR = FN / (TP + FN) = 1 - Sensibilidad
    return {
        "Sensibilidad": sensibilidad,
        "FNR": fnr,
        "Especificidad": especificidad,
        "Precision": precision,
        "F1": f1,
        "Exactitud": exactitud,
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
    }


def main():
    print("=" * 80)
    print("EVALUACIÓN FINAL EN EL CONJUNTO DE PRUEBA RESERVADO (30% = 7,500 REGISTROS)")
    print("=" * 80)

    X_test = pd.read_pickle("data/procesado/X_test.pkl")
    y_test = pd.read_pickle("data/procesado/y_test.pkl")
    df_cv = pd.read_csv("resultados/tabla_validacion_cruzada.csv")

    with open("resultados/mejores_modelos_entrenados.pkl", "rb") as f:
        mejores_modelos = pickle.load(f)

    # 1. Imprimir resumen completo de Validación Cruzada (Media ± Desviación Estándar)
    print("\n1. TABLA RESUMEN DE VALIDACIÓN CRUZADA (5-FOLD: MEDIA ± STD)")
    print("-" * 80)
    tabla_cv_presentacion = []
    for _, row in df_cv.iterrows():
        tabla_cv_presentacion.append(
            {
                "Clasificador": row["Clasificador"],
                "Condicion": row["Condicion"],
                "Sensibilidad (CV)": f"{row['CV_Sensibilidad_Media']*100:.2f}% ± {row['CV_Sensibilidad_Std']*100:.2f}%",
                "FNR (CV)": f"{row['CV_FNR_Media']*100:.2f}% ± {row['CV_FNR_Std']*100:.2f}%",
                "Especificidad (CV)": f"{row['CV_Especificidad_Media']*100:.2f}% ± {row['CV_Especificidad_Std']*100:.2f}%",
                "Precision (CV)": f"{row['CV_Precision_Media']*100:.2f}% ± {row['CV_Precision_Std']*100:.2f}%",
                "F1-Score (CV)": f"{row['CV_F1_Media']*100:.2f}% ± {row['CV_F1_Std']*100:.2f}%",
                "Exactitud (CV)": f"{row['CV_Exactitud_Media']*100:.2f}% ± {row['CV_Exactitud_Std']*100:.2f}%",
            }
        )
    df_cv_fmt = pd.DataFrame(tabla_cv_presentacion)
    print(df_cv_fmt.to_string(index=False))
    df_cv_fmt.to_csv("resultados/tabla_cv_formateada.csv", index=False)

    # 2. Evaluar los 8 modelos en el conjunto de prueba (X_test)
    resultados_test = []
    predicciones_test = {}

    for _, row in df_cv.iterrows():
        config_id = row["Configuracion"]
        modelo = mejores_modelos[config_id]
        y_pred = modelo.predict(X_test)
        predicciones_test[config_id] = y_pred

        m = calcular_metricas(y_test, y_pred)
        m["Configuracion"] = config_id
        m["Clasificador"] = row["Clasificador"]
        m["Condicion"] = row["Condicion"]
        resultados_test.append(m)

    df_test = pd.DataFrame(resultados_test)
    df_test.to_csv("resultados/tabla_evaluacion_prueba.csv", index=False)

    print("\n" + "=" * 80)
    print("2. RESULTADOS EN CONJUNTO DE PRUEBA (7,487 LEGÍTIMAS Y 13 FRAUDES REALES)")
    print("-" * 80)
    cols_mostrar = [
        "Clasificador",
        "Condicion",
        "Sensibilidad",
        "FNR",
        "Especificidad",
        "Precision",
        "F1",
        "Exactitud",
        "TP",
        "FN",
        "FP",
        "TN",
    ]
    df_test_print = df_test[cols_mostrar].copy()
    for c in ["Sensibilidad", "FNR", "Especificidad", "Precision", "F1", "Exactitud"]:
        df_test_print[c] = (df_test_print[c] * 100).map(lambda x: f"{x:.2f}%")
    print(df_test_print.to_string(index=False))

    # Media y Desviación Estándar global por condición en Prueba
    print("\n" + "=" * 80)
    print("3. MEDIA Y DESVIACIÓN ESTÁNDAR ENTRE MODELOS POR CONDICIÓN EN PRUEBA")
    print("-" * 80)
    for cond, grp in df_test.groupby("Condicion"):
        print(f"\nCondición: {cond}")
        for met in ["Sensibilidad", "FNR", "Especificidad", "Precision", "F1", "Exactitud"]:
            print(f"  - {met:<15}: {grp[met].mean()*100:6.2f}% ± {grp[met].std()*100:5.2f}%")

    # 4. Cálculos exactos para responder las preguntas de la Sección 4.7
    print("\n" + "=" * 80)
    print("4. DATOS CLAVE PARA PREGUNTAS DE ANÁLISIS CRÍTICO (SECCIÓN 4.7)")
    print("-" * 80)
    y_dummy_legit = np.zeros_like(y_test)
    m_dummy_0 = calcular_metricas(y_test, y_dummy_legit)
    print(
        f"A) Clasificador que SIEMPRE predice Legítima (Clase 0) en Test:\n"
        f"   - Exactitud: {m_dummy_0['Exactitud']*100:.4f}% | Sensibilidad: {m_dummy_0['Sensibilidad']*100:.2f}% | FNR: {m_dummy_0['FNR']*100:.2f}%\n"
        f"   - Fraudes detectados (TP): {m_dummy_0['TP']} de 13 | Fraudes omitidos (FN): {m_dummy_0['FN']}"
    )

    y_dummy_fraud = np.ones_like(y_test)
    m_dummy_1 = calcular_metricas(y_test, y_dummy_fraud)
    print(
        f"\nB) Clasificador que SIEMPRE predice Fraude (Clase 1) en Test:\n"
        f"   - Exactitud: {m_dummy_1['Exactitud']*100:.4f}% | Sensibilidad: {m_dummy_1['Sensibilidad']*100:.2f}% | Precisión: {m_dummy_1['Precision']*100:.4f}%\n"
        f"   - Falsas alarmas (FP): {m_dummy_1['FP']:,} transacciones legítimas bloqueadas erróneamente."
    )

    # 5. Generar Gráficos para la Presentación
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Matrices de Confusión de las 8 configuraciones
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "Matrices de Confusión en Conjunto de Prueba (30% Reservado: 7,487 Legítimas / 13 Fraudes)",
        fontsize=15,
        fontweight="bold",
    )

    orden_modelos = ["k-NN", "Arbol_Decision", "SVM", "Random_Forest_Propio"]
    conds = ["Sin_Compensacion", "Con_Compensacion_SMOTE"]

    for i, cond in enumerate(conds):
        for j, mod in enumerate(orden_modelos):
            ax = axes[i, j]
            cid = f"{mod}__{cond}"
            cm = confusion_matrix(y_test, predicciones_test[cid], labels=[0, 1])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues" if i == 0 else "Oranges",
                cbar=False,
                ax=ax,
                xticklabels=["Pred: Legít (0)", "Pred: Fraude (1)"],
                yticklabels=["Real: Legít (0)", "Real: Fraude (1)"],
                annot_kws={"size": 12, "weight": "bold"},
            )
            titulo_corto = f"{mod.replace('_', ' ')}\n({'Sin Comp.' if i==0 else 'Con SMOTE'})"
            ax.set_title(titulo_corto, fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("resultados/matrices_confusion_test.png", dpi=300)
    plt.close()

    # Gráfico 2: Comparación de Sensibilidad y F1-Score en Prueba
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    df_plot = df_test.copy()
    df_plot["Sensibilidad (%)"] = df_plot["Sensibilidad"] * 100
    df_plot["Precision (%)"] = df_plot["Precision"] * 100
    df_plot["F1-Score (%)"] = df_plot["F1"] * 100

    sns.barplot(
        data=df_plot,
        x="Clasificador",
        y="Sensibilidad (%)",
        hue="Condicion",
        ax=axes[0],
        palette=["#4C72B0", "#DD8452"],
    )
    axes[0].set_title("Sensibilidad en Prueba (Detección de Fraude - Prioridad)", fontweight="bold")
    axes[0].set_ylim(0, 105)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    sns.barplot(
        data=df_plot,
        x="Clasificador",
        y="F1-Score (%)",
        hue="Condicion",
        ax=axes[1],
        palette=["#4C72B0", "#DD8452"],
    )
    axes[1].set_title("F1-Score en Prueba (Equilibrio Sensibilidad vs Precisión)", fontweight="bold")
    axes[1].set_ylim(0, 105)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig("resultados/grafico_comparativo_metricas.png", dpi=300)
    plt.close()

    print("\n[OK] Gráficos generados exitosamente en 'resultados/':")
    print("  - resultados/matrices_confusion_test.png")
    print("  - resultados/grafico_comparativo_metricas.png")


if __name__ == "__main__":
    main()
