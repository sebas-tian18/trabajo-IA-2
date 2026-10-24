import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from custom_rf import CustomRandomForest

warnings.filterwarnings("ignore")
SEED = 42


def specificity_func(y_true, y_pred):
    """Calcula la Especificidad: TN / (TN + FP), equivalente al recall de la clase 0."""
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def fnr_func(y_true, y_pred):
    """Calcula la Tasa de Falsos Negativos: FNR = FN / (TP + FN) = 1 - Sensibilidad."""
    return 1.0 - recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def criterio_seleccion_desempate(cv_results):
    """
    Regla de la sección 4.5.6:
    Selecciona la configuración que maximice la sensibilidad media de la clase fraude.
    En caso de empate, utiliza la precisión media de la clase fraude como criterio secundario.
    """
    sens_media = np.round(cv_results["mean_test_sensibilidad"], 6)
    prec_media = np.round(cv_results["mean_test_precision"], 6)

    max_sens = np.max(sens_media)
    candidatos = np.where(sens_media == max_sens)[0]

    if len(candidatos) == 1:
        return int(candidatos[0])

    # Desempate por precisión de la clase fraude
    mejor_entre_empatados = candidatos[np.argmax(prec_media[candidatos])]
    return int(mejor_entre_empatados)


def main():
    print("=" * 75)
    print("ENTRENAMIENTO, SELECCIÓN DE CARACTERÍSTICAS Y VALIDACIÓN CRUZADA (5-FOLD)")
    print("=" * 75)

    X_train = pd.read_pickle("data/procesado/X_train.pkl")
    y_train = pd.read_pickle("data/procesado/y_train.pkl")

    # 1. Documentar efecto de la estrategia de compensación (Sección 4.4)
    # Mostramos cómo cambia un fold de entrenamiento (80% de X_train) y el X_train completo
    smote_demo = SMOTE(sampling_strategy=0.20, random_state=SEED, k_neighbors=5)
    X_res_full, y_res_full = smote_demo.fit_resample(X_train, y_train)

    print("\n[A] DOCUMENTACIÓN DEL TRATAMIENTO DEL DESBALANCE (SMOTE sampling_strategy=0.20):")
    print(f"  * En X_train completo (antes de SMOTE)  -> Legítimas (0): {(y_train==0).sum():,} | Fraudes (1): {(y_train==1).sum():,}")
    print(f"  * En X_train completo (después de SMOTE)-> Legítimas (0): {(y_res_full==0).sum():,} | Fraudes (1): {(y_res_full==1).sum():,}")
    print("  * En cada Fold de entrenamiento (80% de 17,500 = 14,000 filas):")
    print("    - Antes de SMOTE   : ~13,977 Legítimas (0) y ~23 Fraudes (1)")
    print("    - Después de SMOTE : ~13,977 Legítimas (0) y ~2,795 Fraudes sintéticos/reales (1)")

    # 2. Definir las 5 particiones estratificadas fijas para TODOS los modelos (Sección 4.5.2 y 4.5.3)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 3. Diccionario de métricas para registrar media y desviación estándar en CV
    scoring = {
        "sensibilidad": make_scorer(recall_score, pos_label=1, zero_division=0),
        "especificidad": make_scorer(specificity_func),
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "exactitud": make_scorer(accuracy_score),
        "fnr": make_scorer(fnr_func),
    }

    # 4. Definición de los 4 clasificadores y sus espacios de búsqueda acotados (Sección 4.5.5)
    modelos_base = {
        "k-NN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7, 11],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        ),
        "Arbol_Decision": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [4, 6, 8, 12, None],
                "clf__min_samples_split": [2, 10],
            },
        ),
        "SVM": (
            SVC(random_state=SEED),
            {
                "clf__C": [0.5, 1.0, 10.0],
                "clf__kernel": ["linear", "rbf"],
                "clf__gamma": ["scale"],
            },
        ),
        "Random_Forest_Propio": (
            CustomRandomForest(random_state=SEED),
            {
                "clf__n_estimators": [25, 51],
                "clf__max_depth": [6, 10, None],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
    }

    condiciones = [
        ("Sin_Compensacion", None),
        (
            "Con_Compensacion_SMOTE",
            SMOTE(sampling_strategy=0.20, random_state=SEED, k_neighbors=5),
        ),
    ]

    resultados_cv = []
    mejores_modelos = {}
    caracteristicas_por_config = {}

    print("\n[B] INICIANDO BÚSQUEDA DE HIPERPARÁMETROS (8 CONFIGURACIONES EXPERIMENTALES)...")
    print("-" * 75)

    for nombre_modelo, (instancia_clf, param_grid) in modelos_base.items():
        for nombre_cond, paso_balanceo in condiciones:
            config_id = f"{nombre_modelo}__{nombre_cond}"
            t0 = time.time()

            # Construcción del Pipeline sin Fuga de Información (Sección 4.2, 4.4 y 4.5)
            selector = SelectFromModel(
                ExtraTreesClassifier(n_estimators=50, random_state=SEED),
                max_features=15,
                threshold=-np.inf,  # Selecciona exactamente las 15 variables más importantes en cada fold
            )

            pasos = [
                ("scaler", StandardScaler()),
                ("feature_selection", selector),
            ]
            if paso_balanceo is not None:
                pasos.append(("balance", paso_balanceo))
            pasos.append(("clf", instancia_clf))

            pipeline = ImbPipeline(pasos)

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit=criterio_seleccion_desempate,
                cv=skf,
                n_jobs=-1,
                 error_score=0.0,
            )

            grid.fit(X_train, y_train)
            duracion = time.time() - t0

            best_idx = grid.best_index_
            cv_res = grid.cv_results_

            # Extraer las características seleccionadas al reajustar en el conjunto de entrenamiento (Sección 4.2)
            mask_features = grid.best_estimator_.named_steps["feature_selection"].get_support()
            cols_seleccionadas = list(X_train.columns[mask_features])
            caracteristicas_por_config[config_id] = cols_seleccionadas

            fila = {
                "Configuracion": config_id,
                "Clasificador": nombre_modelo,
                "Condicion": nombre_cond,
                "Mejores_Hiperparametros": str(grid.best_params_),
                "CV_Sensibilidad_Media": cv_res["mean_test_sensibilidad"][best_idx],
                "CV_Sensibilidad_Std": cv_res["std_test_sensibilidad"][best_idx],
                "CV_FNR_Media": cv_res["mean_test_fnr"][best_idx],
                "CV_FNR_Std": cv_res["std_test_fnr"][best_idx],
                "CV_Especificidad_Media": cv_res["mean_test_especificidad"][best_idx],
                "CV_Especificidad_Std": cv_res["std_test_especificidad"][best_idx],
                "CV_Precision_Media": cv_res["mean_test_precision"][best_idx],
                "CV_Precision_Std": cv_res["std_test_precision"][best_idx],
                "CV_F1_Media": cv_res["mean_test_f1"][best_idx],
                "CV_F1_Std": cv_res["std_test_f1"][best_idx],
                "CV_Exactitud_Media": cv_res["mean_test_exactitud"][best_idx],
                "CV_Exactitud_Std": cv_res["std_test_exactitud"][best_idx],
                "Num_Features": len(cols_seleccionadas),
                "Features_Seleccionadas": ", ".join(cols_seleccionadas),
                "Tiempo_s": round(duracion, 2),
            }
            resultados_cv.append(fila)
            mejores_modelos[config_id] = grid.best_estimator_

            print(
                f"✓ {config_id:<44} | Sensibilidad CV: {fila['CV_Sensibilidad_Media']*100:6.2f}% (±{fila['CV_Sensibilidad_Std']*100:5.2f}%) "
                f"| Precisión CV: {fila['CV_Precision_Media']*100:6.2f}% | ({duracion:.1f}s)"
            )
            print(f"    Hiperparámetros : {grid.best_params_}")

    df_cv = pd.DataFrame(resultados_cv)
    os.makedirs("resultados", exist_ok=True)
    df_cv.to_csv("resultados/tabla_validacion_cruzada.csv", index=False)

    with open("resultados/mejores_modelos_entrenados.pkl", "wb") as f:
        pickle.dump(mejores_modelos, f)

    print("\n" + "=" * 75)
    print("[C] CARACTERÍSTICAS SELECCIONADAS EN ENTRENAMIENTO (Top 15 por Importancia):")
    print("=" * 75)
    print(f"Variables elegidas ({len(cols_seleccionadas)}): {cols_seleccionadas}")
    print("\n[OK] Resultados de Validación Cruzada guardados en 'resultados/tabla_validacion_cruzada.csv'")
    print("[OK] Modelos reentrenados en X_train guardados en 'resultados/mejores_modelos_entrenados.pkl'")


if __name__ == "__main__":
    main()
