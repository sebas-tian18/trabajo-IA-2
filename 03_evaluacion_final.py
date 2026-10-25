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
N_BOOTSTRAP = 1000  # réplicas bootstrap del conjunto de prueba (Sección 4.6)
METRICAS = ["Sensibilidad", "Especificidad", "Precision", "F1", "Exactitud", "FNR"]


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


def metricas_desde_conteos(tp, fn, fp, tn):
    """Calcula las 6 métricas a partir de la matriz de confusión (acepta arreglos numpy)."""
    tp, fn, fp, tn = (np.asarray(v, dtype=float) for v in (tp, fn, fp, tn))
    sens = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    espec = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)  # 0 si no predice fraudes
    f1 = np.divide(2 * prec * sens, prec + sens, out=np.zeros_like(tp), where=(prec + sens) > 0)
    exac = (tp + tn) / (tp + fn + fp + tn)
    return {"Sensibilidad": sens, "Especificidad": espec, "Precision": prec,
            "F1": f1, "Exactitud": exac, "FNR": 1.0 - sens}


def bootstrap_prueba(y_true, predicciones, n_rep=N_BOOTSTRAP, seed=SEED):
    """
    Estima media y desviación estándar de cada métrica en PRUEBA mediante bootstrap estratificado:
    en cada réplica se remuestrean con reemplazo los fraudes y las legítimas por separado,
    manteniendo las cantidades originales (13 y 7.487), es decir, la distribución real de clases.
    Los modelos NO se reentrenan ni se ajustan: solo se reevalúan sus predicciones ya hechas.
    Se usan los MISMOS índices de remuestreo para todos los modelos (comparación pareada).
    """
    y_true = np.asarray(y_true)
    rng = np.random.RandomState(seed)
    idx_fraude = np.where(y_true == 1)[0]
    idx_legit = np.where(y_true == 0)[0]
    muestras = np.array([
        np.concatenate([
            rng.choice(idx_fraude, size=len(idx_fraude), replace=True),
            rng.choice(idx_legit, size=len(idx_legit), replace=True),
        ])
        for _ in range(n_rep)
    ])  # forma (n_rep, n_prueba)
    yb = y_true[muestras]

    filas = []
    for config_id, y_pred in predicciones.items():
        pb = np.asarray(y_pred)[muestras]
        tp = ((yb == 1) & (pb == 1)).sum(axis=1)
        fn = ((yb == 1) & (pb == 0)).sum(axis=1)
        fp = ((yb == 0) & (pb == 1)).sum(axis=1)
        tn = ((yb == 0) & (pb == 0)).sum(axis=1)
        m = metricas_desde_conteos(tp, fn, fp, tn)
        fila = {"Configuracion": config_id}
        for met in METRICAS:
            fila[f"{met}_Media"] = m[met].mean()
            fila[f"{met}_Std"] = m[met].std(ddof=1)
        filas.append(fila)
    return pd.DataFrame(filas)


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

    # 3. Media y desviación estándar POR MODELO Y CONDICIÓN en prueba (Sección 4.6)
    print("\n" + "=" * 80)
    print(f"3. PRUEBA: MEDIA ± DESVIACIÓN ESTÁNDAR POR MODELO (BOOTSTRAP ESTRATIFICADO, {N_BOOTSTRAP} RÉPLICAS)")
    print("-" * 80)
    df_boot = bootstrap_prueba(y_test, predicciones_test)
    df_boot = df_cv[["Configuracion", "Clasificador", "Condicion"]].merge(df_boot, on="Configuracion")
    df_boot.to_csv("resultados/tabla_prueba_media_std.csv", index=False)

    tabla_boot_fmt = df_boot[["Clasificador", "Condicion"]].copy()
    for met in METRICAS:
        tabla_boot_fmt[met] = [
            f"{m*100:.2f}% ± {s*100:.2f}%"
            for m, s in zip(df_boot[f"{met}_Media"], df_boot[f"{met}_Std"])
        ]
    print(tabla_boot_fmt.to_string(index=False))
    tabla_boot_fmt.to_csv("resultados/tabla_prueba_media_std_formateada.csv", index=False)

    # 3b. Desempeño en prueba de la configuración elegida por validación cruzada (pregunta 4.7.4)
    df_sel = pd.read_csv("resultados/configuracion_seleccionada.csv")
    cid_sel = df_sel.loc[0, "Configuracion"]
    puntual = df_test.set_index("Configuracion").loc[cid_sel]
    boot = df_boot.set_index("Configuracion").loc[cid_sel]
    print("\n" + "=" * 80)
    print("3b. CONFIGURACIÓN SELECCIONADA POR VALIDACIÓN CRUZADA Y SU DESEMPEÑO EN PRUEBA")
    print("-" * 80)
    print(f"  Configuración   : {cid_sel}")
    print(f"  Hiperparámetros : {df_sel.loc[0, 'Mejores_Hiperparametros']}")
    print(f"  Sensibilidad CV : {df_sel.loc[0, 'CV_Sensibilidad_Media']*100:.2f}% ± {df_sel.loc[0, 'CV_Sensibilidad_Std']*100:.2f}%")
    print(f"  Prueba (puntual): TP={puntual['TP']}  FN={puntual['FN']}  FP={puntual['FP']}  TN={puntual['TN']}")
    for met in METRICAS:
        print(f"  {met:<15} prueba: {puntual[met]*100:6.2f}%  (bootstrap: {boot[f'{met}_Media']*100:.2f}% ± {boot[f'{met}_Std']*100:.2f}%)")
    print("  Nota: la elección se hizo solo con CV; los resultados de prueba no modifican la decisión.")

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

    # Gráfico 2: Sensibilidad y F1-Score en prueba por modelo y condición.
    # Altura de la barra = valor puntual en prueba; línea negra = ± 1 desviación estándar (bootstrap).
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colores = {"Sin_Compensacion": "#4C72B0", "Con_Compensacion_SMOTE": "#DD8452"}
    etiquetas = {"Sin_Compensacion": "Sin compensación", "Con_Compensacion_SMOTE": "Con SMOTE"}
    x = np.arange(len(orden_modelos))
    ancho = 0.38
    for ax, met, titulo in [
        (axes[0], "Sensibilidad", "Sensibilidad en Prueba (Detección de Fraude - Prioridad)"),
        (axes[1], "F1", "F1-Score en Prueba (Equilibrio Sensibilidad vs Precisión)"),
    ]:
        for k, cond in enumerate(conds):
            ids = [f"{mod}__{cond}" for mod in orden_modelos]
            valores = df_test.set_index("Configuracion").loc[ids, met].values * 100
            errores = df_boot.set_index("Configuracion").loc[ids, f"{met}_Std"].values * 100
            barras = ax.bar(x + (k - 0.5) * ancho, valores, ancho, yerr=errores, capsize=4,
                            color=colores[cond], label=etiquetas[cond], error_kw={"elinewidth": 1})
            ax.bar_label(barras, fmt="%.1f%%", padding=3, fontsize=9, label_type="center", color="white")
        ax.set_xticks(x, [m.replace("_", " ") for m in orden_modelos])
        ax.set_ylim(0, 110)
        ax.set_ylabel(f"{met} (%)")
        ax.set_title(titulo, fontweight="bold")
        ax.legend(title="Condición")
    fig.text(0.5, 0.01, f"Barras de error: ± 1 desviación estándar estimada con bootstrap estratificado ({N_BOOTSTRAP} réplicas) del conjunto de prueba.",
             ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("resultados/grafico_comparativo_metricas.png", dpi=300)
    plt.close()

    print("\n[OK] Gráficos generados exitosamente en 'resultados/':")
    print("  - resultados/matrices_confusion_test.png")
    print("  - resultados/grafico_comparativo_metricas.png")
    print("[OK] Tablas de prueba (media ± std por modelo) en 'resultados/tabla_prueba_media_std*.csv'")


if __name__ == "__main__":
    main()