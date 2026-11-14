"""
Tests de las funciones objetivo no supervisadas.

Ejecutar desde la raiz del repositorio:
    PYTHONPATH=src pytest tests/ -v
"""

import numpy as np
import pytest

from objectives import (
    DIRECTION, IForestDetector, LOFDetector, compute_objective,
    objective_davies_bouldin, objective_silhouette, objective_stability,
    penalty_for,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.5, 0.05, size=(400, 10))
    outliers = rng.normal(0.9, 0.05, size=(20, 10))
    return normal.astype(np.float32), np.vstack([normal[:100], outliers]).astype(np.float32)


def test_directions_declaradas():
    """Toda metrica debe declarar su direccion de optimizacion."""
    assert set(DIRECTION) == {"silhouette", "davies_bouldin", "reconstruction", "stability"}
    assert all(v in ("maximize", "minimize") for v in DIRECTION.values())


def test_objetivos_no_reciben_etiquetas():
    """Contrato central del estudio: la firma no admite y_true."""
    import inspect
    for fn in (objective_silhouette, objective_davies_bouldin):
        params = list(inspect.signature(fn).parameters)
        assert "y" not in params and "y_true" not in params


def test_silhouette_en_rango(data):
    train, val = data
    det = IForestDetector(n_estimators=50, contamination=0.05, random_state=42).fit(train)
    v = objective_silhouette(det, val)
    assert -1.0 <= v <= 1.0


def test_davies_bouldin_no_negativo(data):
    train, val = data
    det = IForestDetector(n_estimators=50, contamination=0.05, random_state=42).fit(train)
    v = objective_davies_bouldin(det, val)
    assert v >= 0.0


def test_estabilidad_en_rango(data):
    train, val = data
    det = IForestDetector(n_estimators=100, contamination=0.05, random_state=42).fit(train)
    v = objective_stability(det, val, n_replicas=4, sigma=0.01)
    assert -1.0 <= v <= 1.0


def test_penalizacion_si_un_solo_grupo(data):
    """Contamination minima -> ninguna anomalia -> particion degenerada."""
    train, val = data

    class SinOutliers(IForestDetector):
        def predict_outliers(self, X):
            return np.zeros(len(X), dtype=int)

    det = SinOutliers(n_estimators=50, random_state=42).fit(train)
    assert objective_silhouette(det, val) == penalty_for("silhouette")
    assert objective_davies_bouldin(det, val) == penalty_for("davies_bouldin")


def test_lof_modo_novelty(data):
    """LOF debe poder puntuar datos no vistos durante el ajuste."""
    train, val = data
    det = LOFDetector(n_neighbors=20, contamination=0.05).fit(train)
    s = det.scores(val)
    assert s.shape == (len(val),)


def test_compute_objective_metrica_desconocida(data):
    train, val = data
    det = IForestDetector(n_estimators=20, random_state=42).fit(train)
    with pytest.raises(ValueError):
        compute_objective("no_existe", det, val)
