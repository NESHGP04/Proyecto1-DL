# Proyecto 1 — Competencia de Modelación (MLP)

**CC3092 · Deep Learning y Sistemas Inteligentes**

Predicción del precio de venta de viviendas (`SalePrice`) del dataset **Ames Housing**
mediante un **Multi-Layer Perceptron (MLP)** en PyTorch. La métrica objetivo es el **RMSE**.

El modelo se entrena sobre `log1p(SalePrice)` (justificado en el EDA por la fuerte asimetría del
precio) y se reporta el RMSE en espacio log y en dólares.

## Resultados

| Modelo | RMSE (log) | RMSE ($) |
|---|---|---|
| Ridge (baseline lineal, solo contexto) | 0.117 | — |
| MLP v1 — ensemble 5-fold | 0.1341 | $25,244 |
| **MLP v2 — target encoding + ensemble 5×3** | **0.1115** | **$20,444** |

RMSE reportado como *out-of-fold* (validación cruzada de 5 folds), la estimación más honesta de
generalización. El modelo desplegable para la competencia es **v2**.

## Estructura del repositorio

```
Proyecto1-DL/
├── train.csv                     # Dataset de entrenamiento
├── notebooks/
│   ├── 01_EDA.ipynb              # Análisis exploratorio (sección 2.1 del informe)
│   └── 02_MLP_modelo.ipynb       # Metodología, iteraciones, discusión (2.2–2.5)
├── src/
│   ├── preprocessing.py          # Pipeline de preprocesamiento (fit en train, idéntico en test)
│   └── model.py                  # Definición del MLP + rutina de entrenamiento
├── models/                       # Artefactos entrenados (generados por el notebook 02)
│   ├── preprocessor.joblib       #   preprocesador v1
│   ├── mlp_ensemble.pt           #   ensemble v1 (5 modelos)
│   ├── preprocessor_v2.joblib    #   preprocesador v2
│   └── mlp_ensemble_v2.pt        #   ensemble v2 (15 modelos + mapas de target encoding)
├── predict.py                    # Predicción para el día de la competencia (usa v2 automáticamente)
└── README.md
```

## Requisitos e instalación

Requiere **Python 3.9+**. Se recomienda un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter nbformat ipykernel
```

Para usar el kernel en los notebooks:

```bash
python -m ipykernel install --user --name proyecto1 --display-name "Python 3 (Proyecto1)"
```

## Cómo reproducir los resultados

### 1. Análisis exploratorio (EDA)

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_EDA.ipynb
```

Cubre dimensiones y tipos de variables, estadísticas descriptivas, tratamiento de nulos y outliers,
visualizaciones (distribuciones, correlaciones, relación de cada feature con el target) y las
decisiones de preprocesamiento que alimentan el modelo.

### 2. Entrenamiento del modelo

```bash
jupyter notebook notebooks/02_MLP_modelo.ipynb
```

Ejecuta el notebook de arriba a abajo (kernel *Python 3 (Proyecto1)*). Al terminar regenera los
artefactos en `models/`. Contiene:

- La metodología (arquitectura, split de datos, pérdida, optimizador, regularización).
- El historial de **iteraciones** con su tabla y curvas de entrenamiento.
- La discusión de resultados y el análisis de errores (residuos).
- El entrenamiento y guardado del ensemble final (v1 y v2).

> Tiempo aproximado: ~10 min en CPU/MPS de una Mac.

### 3. Predicción sobre un nuevo dataset (día de la competencia)

```bash
source .venv/bin/activate
python predict.py --test ruta/al/test.csv --out predicciones.csv
```

`predict.py` carga el preprocesador y el ensemble guardados, aplica **exactamente** el mismo
preprocesamiento al set de prueba, promedia las predicciones de los modelos y escribe
`predicciones.csv` (`Id`, `SalePrice_pred`). Si el CSV de prueba incluye la columna `SalePrice`,
imprime además el RMSE (en log y en dólares). Usa el modelo **v2** si está presente; si no, cae a v1.

## Metodología (resumen)

- **Preprocesamiento** (derivado del EDA): imputación semántica de nulos (`NaN` = "no tiene" →
  categoría `None`/0), `LotFrontage` por mediana de barrio, codificación **ordinal** de variables de
  calidad, `log1p` en features sesgadas, estandarización de numéricas y one-hot de nominales (con
  colapso de categorías raras). En v2, **target encoding** de categóricas de alta cardinalidad
  (`Neighborhood`, `Exterior1st/2nd`), calculado dentro de cada fold para evitar *data leakage*.
- **Arquitectura:** MLP 256→128, activación ReLU, dropout 0.2.
- **Entrenamiento:** optimizador Adam, pérdida Huber (`SmoothL1`), `weight_decay` 3e-4,
  `ReduceLROnPlateau` como scheduler y *early stopping*.
- **Validación:** 5-fold *out-of-fold* + **ensemble** (promedio de modelos) para reducir varianza.

## Notas

- Los artefactos de `models/` se versionan para poder predecir sin reentrenar. Si reentrenás el
  notebook, se sobrescriben.
- El entorno `.venv/` y los `__pycache__/` están excluidos vía `.gitignore`.
