"""
objectives.py — detectores y funciones objetivo NO supervisadas.

Ninguna funcion de este modulo recibe etiquetas.

Regla de decision (umbral)
--------------------------
Todos los detectores entregan un puntaje s(x) (mayor = mas anomalo). La
decision es  s(x) > tau,  con  tau = percentil q de los puntajes de un bloque
de CALIBRACION de trafico normal (mitad del bloque `val`, ver common.split_val).
q se fija a priori (config threshold.q) y es igual para todos los detectores y
escenarios: el umbral deja de ser un hiperparametro que el optimizador pueda
manipular. Esto elimina el optimo degenerado de Silhouette / Davies-Bouldin
(marcar ~0 % de las muestras como anomalas).

Metricas
--------
    silhouette      maximizar   (indice sobre el espacio de puntajes, 2 grupos)
    davies_bouldin  minimizar   (idem)
    reconstruction  minimizar   (error medio de reconstruccion; natural en AE)
    stability       maximizar   (Spearman medio entre replicas con ruido)
    consistency     maximizar   (Spearman medio entre modelos reajustados con
                                 submuestras distintas del entrenamiento)

Autoencoder en el borde
-----------------------
El AE se entrena con Keras (si hay TensorFlow) o con scikit-learn, pero la
INFERENCIA es siempre una pasada hacia adelante en NumPy con los pesos
exportados. Asi la Raspberry Pi no necesita TensorFlow y se evita el costo fijo
de `model.predict` (~50 ms por muestra en los resultados previos).
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor

DIRECTION = {
    "silhouette": "maximize",
    "davies_bouldin": "minimize",
    "reconstruction": "minimize",
    "stability": "maximize",
    "consistency": "maximize",
    "oracle": "maximize",       # MCC con etiquetas de test_oracle, q fijo
    "oracle_q": "maximize",     # MCC con etiquetas de test_oracle, q tambien se busca
}
PENALTY = {"maximize": -1.0, "minimize": 1e6}


def penalty_for(metric: str) -> float:
    return PENALTY[DIRECTION[metric]]


# --------------------------------------------------------------------------- #
# Detectores
# --------------------------------------------------------------------------- #
class BaseDetector:
    name = ""
    q = 0.95
    threshold_ = None
    calib_scores_ = None

    def scores(self, X: np.ndarray) -> np.ndarray:  # mayor = mas anomalo
        raise NotImplementedError

    def set_threshold(self, X_calib: np.ndarray, q: float | None = None):
        if q is not None:
            self.q = float(q)
        self.calib_scores_ = np.sort(np.asarray(self.scores(X_calib), dtype=np.float64))
        self.threshold_ = float(np.quantile(self.calib_scores_, self.q))
        return self

    def threshold_for(self, q: float) -> float:
        return float(np.quantile(self.calib_scores_, q))

    def predict_outliers(self, X: np.ndarray, q: float | None = None) -> np.ndarray:
        thr = self.threshold_ if q is None else self.threshold_for(q)
        return (self.scores(X) > thr).astype(int)

    def predict_native(self, X: np.ndarray) -> np.ndarray:
        """Decision que toma la biblioteca por defecto (solo para la linea base nativa)."""
        raise NotImplementedError


class IForestDetector(BaseDetector):
    name = "iforest"

    def __init__(self, **params):
        params = dict(params)
        params.pop("contamination", None)        # el umbral es el percentil q
        self.random_state = params.pop("random_state", 42)
        self.params = params

    def fit(self, X: np.ndarray) -> "IForestDetector":
        p = dict(self.params)
        ms = p.get("max_samples", "auto")
        if isinstance(ms, (int, np.integer)):
            p["max_samples"] = int(min(ms, len(X)))
        self.model = IsolationForest(random_state=self.random_state, n_jobs=1, **p)
        self.model.fit(X)
        return self

    def scores(self, X):
        return -self.model.score_samples(X)

    def predict_native(self, X):
        return (self.model.predict(X) == -1).astype(int)


class LOFDetector(BaseDetector):
    """LOF en modo novelty. Guarda TODO el train: el costo en RAM y latencia
    crece con su tamano (por eso common.train_for lo acota)."""
    name = "lof"

    def __init__(self, **params):
        params = dict(params)
        params.pop("contamination", None)
        params.pop("random_state", None)
        self.params = params

    def fit(self, X):
        self.model = LocalOutlierFactor(novelty=True, n_jobs=1, **self.params)
        self.model.fit(X)
        return self

    def scores(self, X):
        return -self.model.score_samples(X)

    def predict_native(self, X):
        return (self.model.predict(X) == -1).astype(int)


def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


class AutoencoderDetector(BaseDetector):
    """Autoencoder denso entrenado solo con trafico normal. Puntaje = MSE por fila."""
    name = "autoencoder"

    def __init__(self, n_layers=2, units=64, latent_dim=8, learning_rate=1e-3,
                 batch_size=256, dropout=0.0, epochs=30, random_state=42,
                 backend="auto", **_ignored):
        self.cfg = dict(n_layers=n_layers, units=units, latent_dim=latent_dim,
                        learning_rate=learning_rate, batch_size=batch_size,
                        dropout=dropout, epochs=epochs, random_state=random_state)
        self.backend = backend
        self.layers_: list = []
        self.native_threshold_ = None

    def _sizes(self):
        c = self.cfg
        return [max(c["latent_dim"] + 1, int(c["units"] / (2 ** i))) for i in range(c["n_layers"])]

    def _fit_keras(self, X):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        c = self.cfg
        tf.random.set_seed(c["random_state"])
        sizes = self._sizes()
        inp = keras.Input(shape=(X.shape[1],))
        h = inp
        for s in sizes:
            h = layers.Dense(s, activation="relu")(h)
            if c["dropout"] > 0:
                h = layers.Dropout(c["dropout"])(h)
        h = layers.Dense(c["latent_dim"], activation="relu")(h)
        for s in reversed(sizes):
            h = layers.Dense(s, activation="relu")(h)
        out = layers.Dense(X.shape[1], activation="linear")(h)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=c["learning_rate"]), loss="mse")
        model.fit(X, X, epochs=c["epochs"], batch_size=c["batch_size"], shuffle=True, verbose=0)
        w = model.get_weights()                       # [W1, b1, W2, b2, ...]
        self.layers_ = [(w[i].astype(np.float32), w[i + 1].astype(np.float32))
                        for i in range(0, len(w), 2)]

    def _fit_sklearn(self, X):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        c = self.cfg
        sizes = self._sizes()
        mlp = MLPRegressor(hidden_layer_sizes=tuple(sizes + [c["latent_dim"]] + sizes[::-1]),
                           activation="relu", solver="adam",
                           learning_rate_init=c["learning_rate"],
                           batch_size=int(min(c["batch_size"], len(X))),
                           max_iter=int(c["epochs"]), tol=1e-6, shuffle=True,
                           random_state=c["random_state"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.fit(X, X)       # nota: este backend no aplica dropout
        self.layers_ = [(W.astype(np.float32), b.astype(np.float32))
                        for W, b in zip(mlp.coefs_, mlp.intercepts_)]

    def fit(self, X):
        use_keras = self.backend == "keras" or (self.backend == "auto" and _tf_available())
        (self._fit_keras if use_keras else self._fit_sklearn)(X)
        self.native_threshold_ = float(np.percentile(self.scores(X), 95.0))
        return self

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(np.float32, copy=False)
        for i, (W, b) in enumerate(self.layers_):
            h = h @ W + b
            if i < len(self.layers_) - 1:
                np.maximum(h, 0.0, out=h)
        return h

    def scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        for s in range(0, len(X), 8192):
            xb = X[s:s + 8192]
            out[s:s + 8192] = np.mean((xb - self._forward(xb)) ** 2, axis=1)
        return out

    def predict_native(self, X):
        return (self.scores(X) > self.native_threshold_).astype(int)


DETECTORS = {"iforest": IForestDetector, "lof": LOFDetector, "autoencoder": AutoencoderDetector}


def model_size_kb(det) -> float:
    try:
        return len(pickle.dumps(det, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Funciones objetivo (X = bloque `objetivo` de validacion: solo normal)
# --------------------------------------------------------------------------- #
def _score_space(det, X):
    s = np.asarray(det.scores(X), dtype=np.float64).ravel()
    sd = s.std()
    z = (s - s.mean()) / sd if sd > 0 else s * 0.0
    return z.reshape(-1, 1), (s > det.threshold_).astype(int)


def objective_silhouette(det, X, cfg) -> float:
    z, part = _score_space(det, X)
    if len(np.unique(part)) < 2:
        return penalty_for("silhouette")
    n = min(len(z), int(cfg["optimization"]["silhouette_sample"]))
    try:
        return float(silhouette_score(z, part, sample_size=n, random_state=0))
    except ValueError:
        return penalty_for("silhouette")


def objective_davies_bouldin(det, X, cfg) -> float:
    z, part = _score_space(det, X)
    if len(np.unique(part)) < 2:
        return penalty_for("davies_bouldin")
    return float(davies_bouldin_score(z, part))


def objective_reconstruction(det, X, cfg) -> float:
    return float(np.mean(det.scores(X)))


def _mean_pairwise_spearman(vectors) -> float | None:
    corrs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                continue
            rho = spearmanr(vectors[i], vectors[j]).statistic
            if np.isfinite(rho):
                corrs.append(rho)
    return float(np.mean(corrs)) if corrs else None


def objective_stability(det, X, cfg, seed: int = 42) -> float:
    """Estabilidad de ranking: Spearman medio entre los puntajes de n replicas del
    bloque objetivo con ruido gaussiano (sigma en unidades estandarizadas)."""
    o = cfg["objectives"]
    rng = np.random.default_rng(seed)
    reps = [np.asarray(det.scores(X + rng.normal(0.0, o["stability_noise_sigma"], X.shape)
                                  .astype(X.dtype)), dtype=np.float64).ravel()
            for _ in range(int(o["stability_replicas"]))]
    v = _mean_pairwise_spearman(reps)
    return penalty_for("stability") if v is None else v


def objective_consistency(det, X, cfg, ctx, seed: int = 42) -> float:
    """Consistencia bajo remuestreo: se reajusta el MISMO hiperparametrizado sobre
    `consistency_boot` submuestras distintas del train y se promedia el Spearman
    entre los puntajes que cada modelo asigna al bloque objetivo."""
    if ctx is None:
        raise ValueError("consistency requiere ctx={'Detector','params','X_train'}")
    o = cfg["optimization"]
    rng = np.random.default_rng(seed)
    Xtr = ctx["X_train"]
    k = max(int(o["consistency_frac"] * len(Xtr)), 10)
    vecs = []
    for _ in range(int(o["consistency_boot"])):
        p = dict(ctx["params"])
        if "random_state" in p:
            p["random_state"] = int(rng.integers(0, 2**31 - 1))
        d = ctx["Detector"](**p).fit(Xtr[rng.choice(len(Xtr), size=k, replace=False)])
        vecs.append(np.asarray(d.scores(X), dtype=np.float64).ravel())
    v = _mean_pairwise_spearman(vecs)
    return penalty_for("consistency") if v is None else v


UNSUPERVISED = ("silhouette", "davies_bouldin", "reconstruction", "stability", "consistency")


def compute_objective(metric: str, det, X, cfg: Dict[str, Any], ctx=None) -> float:
    if metric == "silhouette":
        return objective_silhouette(det, X, cfg)
    if metric == "davies_bouldin":
        return objective_davies_bouldin(det, X, cfg)
    if metric == "reconstruction":
        return objective_reconstruction(det, X, cfg)
    if metric == "stability":
        return objective_stability(det, X, cfg)
    if metric == "consistency":
        return objective_consistency(det, X, cfg, ctx)
    raise ValueError(f"Metrica no supervisada desconocida: {metric}. Opciones: {UNSUPERVISED}")
