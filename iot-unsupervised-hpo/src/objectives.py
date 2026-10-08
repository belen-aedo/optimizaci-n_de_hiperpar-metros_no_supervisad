"""
objectives.py — Etapa 1 (bloque 4 del diagrama metodologico).

Implementa las cuatro funciones objetivo ESTRICTAMENTE NO SUPERVISADAS que se
comparan en esta investigacion, junto con los envoltorios de los tres
detectores. Ninguna funcion de este modulo recibe etiquetas.

Metricas
--------
    silhouette      (iForest, LOF)  -> maximizar
    davies_bouldin  (iForest, LOF)  -> minimizar
    reconstruction  (Autoencoder)   -> minimizar
    stability       (Autoencoder)   -> maximizar

Transformacion al espacio de puntajes
-------------------------------------
Silhouette y Davies-Bouldin fueron concebidos para evaluar agrupamientos y no
pueden aplicarse a los arboles de iForest ni a las distancias locales de LOF.
Por eso se calculan sobre el vector continuo de puntajes de anomalia z,
estandarizado, con la particion binaria inducida por el umbral de decision del
propio modelo. Esta transformacion es una decision metodologica documentada en
el informe y constituye una limitacion reconocida del estudio.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor

# Direccion de optimizacion de cada metrica
DIRECTION = {
    "silhouette": "maximize",
    "davies_bouldin": "minimize",
    "reconstruction": "minimize",
    "stability": "maximize",
}

# Valor devuelto cuando la configuracion es degenerada (un solo grupo, etc.)
PENALTY = {"maximize": -1.0, "minimize": 1e6}


def penalty_for(metric: str) -> float:
    return PENALTY[DIRECTION[metric]]


# --------------------------------------------------------------------------- #
# Detectores
# --------------------------------------------------------------------------- #
class IForestDetector:
    """Isolation Forest. score_samples: mayor = mas normal."""

    name = "iforest"

    def __init__(self, **params):
        self.params = params
        self.model = IsolationForest(random_state=params.pop("random_state", 42), **params)

    def fit(self, X: np.ndarray) -> "IForestDetector":
        self.model.fit(X)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        # Se invierte el signo para que mayor = mas anomalo
        return -self.model.score_samples(X)

    def predict_outliers(self, X: np.ndarray) -> np.ndarray:
        """1 = anomalia, 0 = normal."""
        return (self.model.predict(X) == -1).astype(int)


class LOFDetector:
    """Local Outlier Factor en modo novelty (permite puntuar datos nuevos)."""

    name = "lof"

    def __init__(self, **params):
        params.pop("random_state", None)
        self.params = params
        self.model = LocalOutlierFactor(novelty=True, **params)

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model.fit(X)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)

    def predict_outliers(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X) == -1).astype(int)


class AutoencoderDetector:
    """
    Autoencoder denso entrenado solo con trafico normal.
    El puntaje de anomalia es el error de reconstruccion por muestra.
    Requiere TensorFlow/Keras.
    """

    name = "autoencoder"

    def __init__(self, n_layers=2, units=64, latent_dim=8, learning_rate=1e-3,
                 batch_size=256, dropout=0.0, epochs=30, random_state=42,
                 threshold_percentile=95.0, verbose=0):
        self.cfg = dict(
            n_layers=n_layers, units=units, latent_dim=latent_dim,
            learning_rate=learning_rate, batch_size=batch_size, dropout=dropout,
            epochs=epochs, random_state=random_state,
            threshold_percentile=threshold_percentile, verbose=verbose,
        )
        self.model = None
        self.threshold_ = None

    def _build(self, n_features: int):
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "El Autoencoder requiere TensorFlow. Instale con:\n"
                "    pip install 'tensorflow>=2.19'\n"
                "Los detectores iForest y LOF no lo necesitan."
            ) from exc

        c = self.cfg
        tf.random.set_seed(c["random_state"])
        sizes = [max(c["latent_dim"] + 1, int(c["units"] / (2 ** i)))
                 for i in range(c["n_layers"])]

        inp = keras.Input(shape=(n_features,))
        h = inp
        for s in sizes:                       # encoder
            h = layers.Dense(s, activation="relu")(h)
            if c["dropout"] > 0:
                h = layers.Dropout(c["dropout"])(h)
        h = layers.Dense(c["latent_dim"], activation="relu", name="latent")(h)
        for s in reversed(sizes):             # decoder
            h = layers.Dense(s, activation="relu")(h)
        out = layers.Dense(n_features, activation="linear")(h)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=c["learning_rate"]),
            loss="mse",
        )
        return model

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        c = self.cfg
        self.model = self._build(X.shape[1])
        self.model.fit(
            X, X, epochs=c["epochs"], batch_size=c["batch_size"],
            shuffle=True, verbose=c["verbose"],
        )
        # Umbral fijado sobre trafico normal de entrenamiento (sin etiquetas)
        self.threshold_ = float(np.percentile(self.scores(X), c["threshold_percentile"]))
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        rec = self.model.predict(X, verbose=0, batch_size=self.cfg["batch_size"])
        return np.mean((X - rec) ** 2, axis=1)

    def predict_outliers(self, X: np.ndarray) -> np.ndarray:
        return (self.scores(X) > self.threshold_).astype(int)


DETECTORS = {
    "iforest": IForestDetector,
    "lof": LOFDetector,
    "autoencoder": AutoencoderDetector,
}


# --------------------------------------------------------------------------- #
# Funciones objetivo
# --------------------------------------------------------------------------- #
def _score_space(detector, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (z, particion) donde z es el puntaje estandarizado en 2D."""
    s = np.asarray(detector.scores(X), dtype=np.float64).ravel()
    sd = s.std()
    z = (s - s.mean()) / sd if sd > 0 else s * 0.0
    # Ambos detectores sklearn usan `offset_` para separar normal/anomalia.
    # Reutilizar los puntajes evita una segunda inferencia por trial.
    uses_default_partition = detector.__class__ in (IForestDetector, LOFDetector)
    if uses_default_partition and hasattr(detector.model, "offset_"):
        part = (s > -float(detector.model.offset_)).astype(int)
    else:
        part = detector.predict_outliers(X)
    return z.reshape(-1, 1), part


def objective_silhouette(detector, X: np.ndarray) -> float:
    z, part = _score_space(detector, X)
    if len(np.unique(part)) < 2 or len(z) < 3:
        return penalty_for("silhouette")
    return float(silhouette_score(z, part))


def objective_davies_bouldin(detector, X: np.ndarray) -> float:
    z, part = _score_space(detector, X)
    if len(np.unique(part)) < 2 or len(z) < 3:
        return penalty_for("davies_bouldin")
    return float(davies_bouldin_score(z, part))


def objective_reconstruction(detector, X: np.ndarray) -> float:
    """Error medio de reconstruccion sobre trafico normal de validacion."""
    return float(np.mean(detector.scores(X)))


def objective_stability(detector, X: np.ndarray, n_replicas: int = 10,
                        sigma: float = 0.01, seed: int = 42) -> float:
    """
    Consistencia del ordenamiento de puntajes ante perturbaciones menores.

    Se generan n_replicas copias del conjunto de validacion con ruido gaussiano
    de desviacion sigma (los datos estan escalados, por lo que sigma opera en
    unidades normalizadas) y se mide la correlacion de Spearman promedio entre
    todos los pares de vectores de puntajes. Las filas se mantienen alineadas
    entre replicas, de modo que la correlacion es interpretable.
    """
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_replicas):
        Xn = X + rng.normal(0.0, sigma, size=X.shape).astype(X.dtype)
        reps.append(np.asarray(detector.scores(Xn), dtype=np.float64).ravel())

    corrs = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            if np.std(reps[i]) == 0 or np.std(reps[j]) == 0:
                continue
            rho = spearmanr(reps[i], reps[j]).statistic
            if np.isfinite(rho):
                corrs.append(rho)
    if not corrs:
        return penalty_for("stability")
    return float(np.mean(corrs))


OBJECTIVES = {
    "silhouette": objective_silhouette,
    "davies_bouldin": objective_davies_bouldin,
    "reconstruction": objective_reconstruction,
    "stability": objective_stability,
}


def compute_objective(metric: str, detector, X: np.ndarray,
                      cfg: Dict[str, Any] | None = None) -> float:
    """Punto de entrada unico usado por optimize.py."""
    if metric not in OBJECTIVES:
        raise ValueError(f"Metrica desconocida: {metric}. Opciones: {list(OBJECTIVES)}")
    if metric == "stability":
        o = (cfg or {}).get("objectives", {})
        return objective_stability(
            detector, X,
            n_replicas=o.get("stability_replicas", 10),
            sigma=o.get("stability_noise_sigma", 0.01),
        )
    return OBJECTIVES[metric](detector, X)
