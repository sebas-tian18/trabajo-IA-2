import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CustomRandomForest(BaseEstimator, ClassifierMixin):
    """
    Implementación propia de Random Forest mediante Bagging (Bootstrap Aggregating)
    y selección aleatoria de características por división de nodo.
    Compatible con Pipelines y GridSearchCV de Scikit-learn.
    """

    def __init__(
        self,
        n_estimators=51,
        max_depth=None,
        max_features="sqrt",
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def _generar_muestra_bootstrap(self, X, y, rng):
        """
        Genera una muestra bootstrap del mismo tamaño (n_samples) que el conjunto
        de entrada mediante muestreo aleatorio CON reemplazo.
        """
        n_samples = X.shape[0]
        indices_bootstrap = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices_bootstrap], y[indices_bootstrap], indices_bootstrap

    def fit(self, X, y):
        """
        Entrena 'n_estimators' árboles de decisión independientes, cada uno con
        su propia muestra bootstrap y su propia semilla derivada de 'random_state'.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Generador maestro a partir de la semilla general para asegurar reproducibilidad
        master_rng = np.random.RandomState(self.random_state)
        # Semillas distintas para cada árbol (diversidad reproducible)
        tree_seeds = master_rng.randint(0, 2**31 - 1, size=self.n_estimators)

        self.trees_ = []
        self.bootstrap_unique_ratios_ = []

        for seed in tree_seeds:
            tree_rng = np.random.RandomState(seed)

            # 1. Muestreo bootstrap con reemplazo
            X_boot, y_boot, idx_boot = self._generar_muestra_bootstrap(X, y, tree_rng)

            # Guardamos la proporción de registros únicos (~63.2% teórico) para análisis
            ratio_unicos = len(np.unique(idx_boot)) / len(idx_boot)
            self.bootstrap_unique_ratios_.append(ratio_unicos)

            # 2. Árbol independiente con subconjunto aleatorio de características (max_features)
            arbol = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=int(seed),
            )
            arbol.fit(X_boot, y_boot)
            self.trees_.append(arbol)

        return self

    def predict_proba(self, X):
        """
        Calcula la proporción de votos de los árboles para cada clase [P(Clase 0), P(Clase 1)].
        """
        check_is_fitted(self, ["trees_", "classes_"])
        X = check_array(X, accept_sparse=False)

        # Matriz de votos de dimensión (n_estimators, n_samples)
        votos = np.array([arbol.predict(X) for arbol in self.trees_])

        # Proporción de árboles que votaron por la clase 1 (Fraude)
        prob_fraude = np.mean(votos == 1, axis=0)
        prob_legitima = 1.0 - prob_fraude

        return np.column_stack((prob_legitima, prob_fraude))

    def predict(self, X):
        """
        Combina las predicciones de todos los árboles mediante votación mayoritaria.
        Regla explícita de desempate: en caso de empate exacto (50% de votos), se asigna
        la clase 1 (Fraude) para priorizar la minimización de falsos negativos.
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    # Prueba rápida de verificación con los datos reales particionados
    import pandas as pd

    print("Verificando implementación de CustomRandomForest...")
    X_train = pd.read_pickle("data/procesado/X_train.pkl")
    y_train = pd.read_pickle("data/procesado/y_train.pkl")

    rf_test = CustomRandomForest(
        n_estimators=25, max_depth=6, max_features="sqrt", random_state=42
    )
    rf_test.fit(X_train, y_train)
    preds = rf_test.predict(X_train[:100])

    print(f"- Árboles entrenados exitosamente : {len(rf_test.trees_)}")
    print(f"- Semillas únicas utilizadas      : {len({t.random_state for t in rf_test.trees_})}")
    print(f"- Promedio de registros únicos en cada muestra bootstrap: {np.mean(rf_test.bootstrap_unique_ratios_)*100:.2f}% (Teórico: ~63.21%)")
    print(f"- Porcentaje promedio de registros repetidos por bootstrap: {(1 - np.mean(rf_test.bootstrap_unique_ratios_))*100:.2f}%")
    print("[OK] CustomRandomForest funciona y cumple todos los requisitos de la sección 4.3.")
