# Evaluación de métricas no supervisadas para la optimización de hiperparámetros en detectores de anomalías IoT

Pipeline experimental del proyecto de Taller de Investigación.

**Autora:** Belén Aedo Peña — Departamento de Ingeniería en Informática, Universidad de Santiago de Chile
**Profesor guía:** Juan Ignacio Iturbe-Araya

---

## Qué hace este repositorio

Compara cuatro métricas **estrictamente no supervisadas** usadas como función objetivo para optimizar los hiperparámetros de tres detectores de anomalías, y mide el efecto de esa elección tanto en la capacidad de detección como en el consumo de recursos de un dispositivo de borde.

| Detector | Métricas evaluadas |
|---|---|
| Isolation Forest | Silhouette, Davies–Bouldin |
| Local Outlier Factor | Silhouette, Davies–Bouldin |
| Autoencoder | Error de reconstrucción, Estabilidad |

**Pregunta de investigación:** ¿qué métricas no supervisadas permiten obtener el mejor equilibrio entre capacidad de detección y consumo de recursos cuando no se dispone de etiquetas durante el entrenamiento?

**Hipótesis:** la efectividad de una métrica no supervisada depende del algoritmo detector empleado.

---

## Dependencia externa: `preprocessor_v2`

Este repositorio **no** implementa el preprocesamiento. Consume las salidas del pipeline `preprocessor_v2` del laboratorio, disponible en <https://github.com/juaniturbe/preprocessor> y desarrollado por Juan Ignacio Iturbe-Araya.

De ese pipeline se utilizan las etapas 01 a 04, que producen:

```
<ROOT>/Datasets/Parquet/Ton-IoT/<selector>/train/best_features.parquet
<ROOT>/Datasets/Parquet/Ton-IoT/<selector>/test/best_features.parquet
```

La etapa 05 (`05_ml_benchmark.py`) generó además el archivo `ml_benchmark_results.csv` usado como evidencia preliminar en el informe: verifica que el preprocesamiento preserva la información discriminante del tráfico y fija empíricamente el espacio de 40 variables (`l1_lasso`) sobre el que opera este repositorio.

> **Nota metodológica.** `preprocessor_v2` divide los datos 80/20 dejando tráfico mixto en ambos bloques. Ese esquema no corresponde al protocolo antifuga de esta investigación, que exige entrenamiento y validación con **solo tráfico normal**. Por eso `src/prepare_splits.py` reúne ambos bloques y reconstruye la partición en tres conjuntos. Es la primera etapa obligatoria.

---

## Requisitos

- Python 3.10
- Ver `requirements.txt`. TensorFlow solo es necesario para el detector Autoencoder; iForest y LOF funcionan sin él.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducción desde cero

### 1. Configurar rutas

Editar `config/experiment.yaml` y apuntar `preprocessor_root` al `ROOT` usado en `preprocessor_v2` (el mismo valor de su `config.yaml`):

```yaml
paths:
  preprocessor_root: "~/TALLER DE INVESTIGACION"
  dataset_name: "Ton-IoT"
  selector: "l1_lasso"
```

### 2. Reconstruir las particiones (bloque 2 del diagrama)

```bash
python src/prepare_splits.py
```

Genera `data/splits/{train,val,test}.parquet` y un `manifest.json` con el número de filas, la distribución de clases y el **hash SHA-256** de cada bloque.

| Partición | Contenido | Uso |
|---|---|---|
| `train` | solo normal, sin etiquetas | ajuste de los detectores |
| `val` | solo normal, sin etiquetas | cálculo de la función objetivo |
| `test` | normal + ataques, etiquetado | evaluación final únicamente |

### 3. Optimizar (bloques 5 y 6)

```bash
# una combinación
python src/optimize.py --detector iforest --metric silhouette

# todas las combinaciones de config/experiment.yaml
python src/optimize.py --all

# prueba rapida para iterar sin ejecutar todo el protocolo
python src/optimize.py --all --fast
```

`--fast` usa 25 trials y una sola semilla (`42`). No modifica la configuracion
del protocolo completo, que conserva 100 trials y cinco semillas.

Salidas: `outputs/optuna/<detector>_<metrica>_seed<n>.json` con los mejores hiperparámetros, el valor del objetivo y la latencia medida.

### 4. Medir en el dispositivo de borde (bloque 7)

**Este paso se ejecuta en la Raspberry Pi 5**, copiando el repositorio junto con `data/splits/` y `outputs/optuna/`:

```bash
python src/edge_benchmark.py
```

Salida: `outputs/results/edge_benchmark.csv` con latencia mediana y p95, CPU, RAM y tamaño del modelo.

### 5. Evaluación final (bloques 8 y 9)

```bash
python src/evaluate.py
```

Único script autorizado a leer las etiquetas del bloque de prueba. Verifica primero el hash del manifiesto y aborta si el conjunto fue alterado. Evalúa la línea base (hiperparámetros por defecto) y cada configuración optimizada.
En equipos con memoria limitada, como Raspberry Pi 5, se puede reducir el tamaño
de lote de inferencia para disminuir el uso máximo de RAM (a costa de algo más
de tiempo), por ejemplo: `python src/evaluate.py --batch-size 512`. La evaluación
se ejecuta secuencialmente y limita a un hilo los detectores sklearn.

Salidas: `outputs/results/final_evaluation.csv`, `balance.csv` y `hypothesis_check.json`.

### 6. Generar las tablas del informe

```bash
python src/report.py
```

Escribe en `outputs/tables/` los `.tex` que el manuscrito incluye con `\input{}`. Si una etapa aún no se ejecutó, la tabla se genera con guiones y la marca `[Pendiente]`, de modo que el informe nunca contenga números inventados.

### Ejecutar los tests

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Estructura

```
├── config/experiment.yaml     Parámetros del experimento
├── src/
│   ├── common.py              Configuración, E/S, semillas, métricas
│   ├── prepare_splits.py      Etapa 0 — partición antifuga
│   ├── objectives.py          Etapa 1 — funciones objetivo y detectores
│   ├── optimize.py            Etapa 2 — búsqueda con Optuna
│   ├── edge_benchmark.py      Etapa 3 — medición en Raspberry Pi 5
│   ├── evaluate.py            Etapa 4 — evaluación final
│   └── report.py              Etapa 5 — tablas LaTeX
├── tests/test_objectives.py
└── outputs/                   Generado por la ejecución
```

---

## Controles metodológicos implementados

- **Sin etiquetas en la optimización.** Ninguna función de `objectives.py` recibe `y_true`; hay un test que lo verifica sobre la firma de las funciones.
- **Conjunto de prueba aislado.** `evaluate.py` es el único script que lo abre, y comprueba el SHA-256 contra el manifiesto antes de usarlo.
- **Restricción de borde como restricción dura.** Cada métrica se optimiza como objetivo único; los *trials* que exceden el presupuesto de latencia se descartan con `TrialPruned` en vez de penalizarse dentro de la función objetivo, para no mezclar el efecto del criterio no supervisado con el del peso asignado al término de hardware.
- **Transformación al espacio de puntajes.** Silhouette y Davies–Bouldin se calculan sobre el vector de puntajes de anomalía estandarizado, con la partición inducida por el umbral del propio modelo. Es una aproximación documentada y constituye una limitación reconocida del estudio.
- **Semillas fijas** en `numpy`, `random`, los estimadores de scikit-learn y TensorFlow.

## Limitación conocida

Los selectores de características empleados en la etapa preliminar (`f_classif`, `mutual_info_classif`, `chi2`, `l1_lasso`) utilizan las etiquetas para puntuar variables. La selección se realiza una única vez y se aplica de forma idéntica a todas las condiciones experimentales, por lo que no puede favorecer a un criterio de optimización sobre otro, pero constituye una ventaja común no disponible en un despliegue estrictamente ciego.
---

## Declaración de uso de inteligencia artificial

Se utilizó Claude como asistente en las siguientes tareas:

- Estructuración del repositorio y redacción inicial de los scripts de `src/`
- Formalización matemática de las cuatro funciones objetivo no supervisadas.
- Generación de las tablas y figuras a partir de los resultados de ejecución.
