import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, confusion_matrix

warnings.filterwarnings("ignore")
SEED = 42

# Colores (validados para daltonismo): uno por entidad, iguales en todos los gráficos de este script
COLOR_LEGITIMA = "#2F6DB5"
COLOR_FRAUDE = "#D1495B"
COLOR_ARBOL = "#7B5EA7"
COLOR_RF = "#2A9D8F"
COLOR_GRIS = "#B0B0B0"


def falsos_positivos(y_true, y_pred):
    """Cantidad de falsas alarmas (legítimas clasificadas como fraude)."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1]


SCORING = {
    "sensibilidad": make_scorer(recall_score, pos_label=1, zero_division=0),
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "falsos_positivos": make_scorer(falsos_positivos),
}


def estilo_ejes(ax):
    """Ejes y grilla discretos para que destaquen los datos."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)


def grafico_exploracion():
    """Gráfico 1 (diapositiva 2): desbalance de clases y montos por clase en el dataset completo."""
    df = pd.read_csv("data/creditcard.csv").drop_duplicates()
    conteo = df["Class"].value_counts().sort_index()
    pct = conteo / conteo.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Distribución de clases en escala lineal: el desbalance se ve tal como es
    ax = axes[0]
    barras = ax.bar(["Legítima (0)", "Fraude (1)"], conteo.values,
                    color=[COLOR_LEGITIMA, COLOR_FRAUDE], width=0.55)
    for barra, n, p in zip(barras, conteo.values, pct.values):
        ax.annotate(f"{n:,}\n({p:.2f}%)".replace(",", "."), (barra.get_x() + barra.get_width() / 2, n),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=11)
    ax.set_ylim(0, conteo.max() * 1.18)
    ax.set_ylabel("Número de transacciones")
    ax.set_title("Distribución de clases (sin duplicados)", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    estilo_ejes(ax)

    # (b) Monto por clase en escala logarítmica (los montos van de 0 a más de 25.000)
    ax = axes[1]
    datos = [df.loc[df["Class"] == 0, "Amount"], df.loc[df["Class"] == 1, "Amount"]]
    caja = ax.boxplot(datos, tick_labels=["Legítima (0)", "Fraude (1)"], patch_artist=True,
                      widths=0.5, showfliers=False, medianprops={"color": "black"})
    for parche, color in zip(caja["boxes"], [COLOR_LEGITIMA, COLOR_FRAUDE]):
        parche.set_facecolor(color)
        parche.set_alpha(0.85)
    for i, serie in enumerate(datos, start=1):
        ax.annotate(f"mediana: {serie.median():.2f}", (i + 0.28, serie.median()), va="center", fontsize=10)
    ax.set_xlim(0.5, 2.9)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Monto de la transacción (escala log)")
    ax.set_title("Monto por clase (sin valores atípicos)", fontweight="bold")
    estilo_ejes(ax)

    fig.text(0.5, 0.01, "Dataset completo tras eliminar 1.081 duplicados (283.726 transacciones). "
             "Gráfico descriptivo: ninguna decisión del modelo se toma a partir de él.", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("resultados/exploracion_datos.png", dpi=300)
    plt.close()

    print("1. EXPLORACIÓN (dataset sin duplicados)")
    print(f"   Legítimas: {conteo[0]:,} ({pct[0]:.2f}%) | Fraudes: {conteo[1]:,} ({pct[1]:.2f}%)")
    print(f"   Proporción: 1 fraude cada {conteo[0] / conteo[1]:.0f} transacciones legítimas")
    print(f"   Mediana del monto -> legítimas: {datos[0].median():.2f} | fraudes: {datos[1].median():.2f}")


def grafico_importancia(mejores_modelos, columnas):
    """Gráfico 2 (metodología): importancia de cada variable según el selector ajustado en X_train."""
    selector = mejores_modelos["Arbol_Decision__Sin_Compensacion"].named_steps["feature_selection"]
    imp = pd.Series(selector.estimator_.feature_importances_, index=columnas).sort_values()
    elegidas = set(np.array(columnas)[selector.get_support()])

    fig, ax = plt.subplots(figsize=(9, 9))
    colores = [COLOR_RF if c in elegidas else COLOR_GRIS for c in imp.index]
    ax.barh(imp.index, imp.values * 100, color=colores, height=0.7)
    ax.set_xlabel("Importancia (% del total, ExtraTrees)")
    ax.set_title("Selección de características: 15 de 30 variables", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    suma = imp[list(elegidas)].sum() * 100
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR_RF), plt.Rectangle((0, 0), 1, 1, color=COLOR_GRIS)],
              labels=[f"Seleccionadas (15) - {suma:.1f}% de la importancia", "Descartadas (15)"],
              loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig("resultados/importancia_variables.png", dpi=300)
    plt.close()

    print("\n2. IMPORTANCIA DE VARIABLES (selector ajustado en X_train)")
    print(f"   Las 15 seleccionadas concentran {suma:.1f}% de la importancia total")
    print(f"   Más importantes: {', '.join(imp.sort_values(ascending=False).index[:5])}")


def comparacion_arbol_vs_rf(mejores_modelos, X_train, y_train, skf):
    """Gráfico 3 (pregunta 4.7.3): árbol individual vs Random Forest propio, fold por fold en CV."""
    filas = []
    for cond in ["Sin_Compensacion", "Con_Compensacion_SMOTE"]:
        for modelo, nombre in [("Arbol_Decision", "Árbol"), ("Random_Forest_Propio", "Random Forest")]:
            # Se reevalúa la configuración ya elegida con las MISMAS 5 particiones (no se ajusta nada nuevo)
            pipeline = clone(mejores_modelos[f"{modelo}__{cond}"])
            res = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=SCORING, n_jobs=-1)
            for fold in range(skf.get_n_splits()):
                filas.append({
                    "Condicion": cond, "Modelo": nombre, "Fold": fold + 1,
                    "Sensibilidad": res["test_sensibilidad"][fold],
                    "Precision": res["test_precision"][fold],
                    "Falsos_Positivos": int(res["test_falsos_positivos"][fold]),
                })
    df = pd.DataFrame(filas)
    df.to_csv("resultados/tabla_arbol_vs_rf_por_fold.csv", index=False)

    std = lambda s: np.std(s)  # misma desviación estándar que reporta GridSearchCV en el script 02
    resumen = df.groupby(["Condicion", "Modelo"], sort=False).agg(
        Sensibilidad_Media=("Sensibilidad", "mean"), Sensibilidad_Std=("Sensibilidad", std),
        Precision_Media=("Precision", "mean"), Precision_Std=("Precision", std),
        FP_Total_5_Folds=("Falsos_Positivos", "sum"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    metricas = [("Sensibilidad", "Sensibilidad por fold (%)", 100),
                ("Precision", "Precisión por fold (%)", 100),
                ("Falsos_Positivos", "Falsas alarmas por fold", 1)]
    posiciones = {("Sin_Compensacion", "Árbol"): 0, ("Sin_Compensacion", "Random Forest"): 1,
                  ("Con_Compensacion_SMOTE", "Árbol"): 3, ("Con_Compensacion_SMOTE", "Random Forest"): 4}
    rng = np.random.RandomState(SEED)
    for ax, (col, titulo, escala) in zip(axes, metricas):
        for (cond, nombre), x in posiciones.items():
            valores = df[(df["Condicion"] == cond) & (df["Modelo"] == nombre)][col].values * escala
            color = COLOR_ARBOL if nombre == "Árbol" else COLOR_RF
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(valores)), valores, s=60, color=color,
                       edgecolor="white", linewidth=1.5, zorder=3)
            ax.hlines(valores.mean(), x - 0.3, x + 0.3, color="black", linewidth=2, zorder=4)
            ax.annotate(f"{valores.mean():.1f}", (x + 0.32, valores.mean()), va="center", fontsize=9)
        ax.set_xticks([0.5, 3.5], ["Sin compensación", "Con SMOTE"])
        ax.set_xlim(-0.7, 5.2)
        ax.set_title(titulo, fontweight="bold")
        estilo_ejes(ax)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLOR_ARBOL, markersize=8),
                        plt.Line2D([], [], marker="o", ls="", color=COLOR_RF, markersize=8),
                        plt.Line2D([], [], color="black", lw=2)],
               labels=["Árbol de decisión", "Random Forest propio", "Media de los 5 folds"],
               loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.01, "Cada punto es un fold de la validación cruzada (mismas 5 particiones). "
             "Configuraciones elegidas en 02_entrenamiento_cv.py.", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig("resultados/arbol_vs_rf.png", dpi=300)
    plt.close()

    print("\n3. ÁRBOL vs RANDOM FOREST PROPIO (validación cruzada, mismas 5 particiones)")
    tabla = resumen.copy()
    for m in ["Sensibilidad", "Precision"]:
        tabla[m] = [f"{a*100:.2f}% ± {b*100:.2f}%" for a, b in zip(tabla[f"{m}_Media"], tabla[f"{m}_Std"])]
    print(tabla[["Condicion", "Modelo", "Sensibilidad", "Precision", "FP_Total_5_Folds"]].to_string(index=False))
    resumen.to_csv("resultados/tabla_arbol_vs_rf.csv", index=False)


def curva_numero_arboles(mejores_modelos, X_train, y_train, skf):
    """Gráfico 4 (pregunta 4.7.3): qué aporta combinar árboles, variando solo n_estimators."""
    base = mejores_modelos["Random_Forest_Propio__Sin_Compensacion"]
    filas = []
    for n in [1, 5, 11, 25, 51]:  # impares: la votación nunca empata
        pipeline = clone(base).set_params(clf__n_estimators=n)
        res = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=SCORING, n_jobs=-1)
        filas.append({
            "N_Arboles": n,
            "Sensibilidad_Media": res["test_sensibilidad"].mean(), "Sensibilidad_Std": res["test_sensibilidad"].std(),
            "Precision_Media": res["test_precision"].mean(), "Precision_Std": res["test_precision"].std(),
            "FP_Total_5_Folds": int(res["test_falsos_positivos"].sum()),
        })
    df = pd.DataFrame(filas)
    df.to_csv("resultados/tabla_numero_arboles.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, met, titulo in [(axes[0], "Sensibilidad", "Sensibilidad en CV (%)"),
                            (axes[1], "Precision", "Precisión en CV (%)")]:
        media = df[f"{met}_Media"] * 100
        std = df[f"{met}_Std"] * 100
        ax.fill_between(df["N_Arboles"], (media - std).clip(0, 100), (media + std).clip(0, 100),
                        color=COLOR_RF, alpha=0.15, linewidth=0)
        ax.plot(df["N_Arboles"], media, color=COLOR_RF, linewidth=2, marker="o", markersize=8)
        for x, y in zip(df["N_Arboles"], media):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(df["N_Arboles"])
        ax.set_xlabel("Número de árboles del ensamble")
        ax.set_title(titulo, fontweight="bold")
        ax.set_ylim(0, 110)
        estilo_ejes(ax)
    fig.text(0.5, 0.01, "Random Forest propio sin compensación; solo cambia n_estimators. "
             "Banda: ± 1 desviación estándar entre folds.", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("resultados/curva_numero_arboles.png", dpi=300)
    plt.close()

    print("\n4. EFECTO DEL NÚMERO DE ÁRBOLES (Random Forest propio sin compensación, CV)")
    tabla = df.copy()
    for m in ["Sensibilidad", "Precision"]:
        tabla[m] = [f"{a*100:.2f}% ± {b*100:.2f}%" for a, b in zip(tabla[f"{m}_Media"], tabla[f"{m}_Std"])]
    print(tabla[["N_Arboles", "Sensibilidad", "Precision", "FP_Total_5_Folds"]].to_string(index=False))


def main():
    print("=" * 80)
    print("GRÁFICOS COMPLEMENTARIOS PARA LA PRESENTACIÓN")
    print("=" * 80)
    X_train = pd.read_pickle("data/procesado/X_train.pkl")
    y_train = pd.read_pickle("data/procesado/y_train.pkl")
    with open("resultados/mejores_modelos_entrenados.pkl", "rb") as f:
        mejores_modelos = pickle.load(f)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # mismas particiones que 02
    os.makedirs("resultados", exist_ok=True)

    grafico_exploracion()
    grafico_importancia(mejores_modelos, X_train.columns)
    comparacion_arbol_vs_rf(mejores_modelos, X_train, y_train, skf)
    curva_numero_arboles(mejores_modelos, X_train, y_train, skf)

    print("\n[OK] Gráficos generados en 'resultados/':")
    for nombre in ["exploracion_datos.png", "importancia_variables.png", "arbol_vs_rf.png", "curva_numero_arboles.png"]:
        print(f"  - resultados/{nombre}")
    print("[OK] Tablas: tabla_arbol_vs_rf.csv, tabla_arbol_vs_rf_por_fold.csv, tabla_numero_arboles.csv")


if __name__ == "__main__":
    main()