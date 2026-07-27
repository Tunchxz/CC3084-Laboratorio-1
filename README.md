# Laboratorio 1: Series de Tiempo

En este laboratorio se trabaja con los datos históricos de ingreso de viajeros internacionales a
Guatemala.

## Estructura del repositorio

```bash
CC3084-Laboratorio-1/
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb   # Ejercicio 1 — EDA
│   ├── 02_preparacion_series.ipynb      # Ejercicios 2 y 3 — partición y construcción de series
│   ├── 03_serie_total.ipynb             # Ejercicio 4 — serie total
│   ├── 04_series_regiones.ipynb         # Ejercicio 4 — categoría Regiones
│   ├── 05_series_fronteras.ipynb        # Ejercicio 4 — categoría Fronteras
│   └── 06_analisis_comparativo.ipynb    # Ejercicio 5 — comparación y hallazgos
├── src/
│   └── ts_utils.py                      # Pipeline compartido de series de tiempo
├── data/
│   ├── raw/                             # Vacío (reservado)
│   └── processed/                       # Series y resúmenes generados por el notebook 02
├── requirements.txt                     # Dependecias del proyecto
├── Informe.pdf
└── README.md
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota sobre Prophet.** `prophet` usa el backend `cmdstanpy`, que requiere un compilador de C++.
> En Windows hay que instalar el toolchain de CmdStan una sola vez:
>
> ```bash
> python -m cmdstanpy.install_cxx_toolchain
> python -m cmdstanpy.install_cmdstan
> ```
>
> Si Prophet no está disponible, `src/ts_utils.py` lo detecta y continúa sin ese modelo en lugar de fallar.

## Orden de ejecución

Los notebooks son secuenciales: el 02 genera los CSV que consumen el 03, 04 y 05, y el 06 lee los
resúmenes que producen el 04 y el 05.

| # | Notebook | Qué hace | Requiere | Produce |
|---|---|---|---|---|
| 1 | `01_analisis_exploratorio.ipynb` | EDA: calidad de datos, descriptivas, comportamiento temporal, países, regiones, vías, fronteras y atípicos | `data/raw/*.xlsx` | — |
| 2 | `02_preparacion_series.ipynb` | Partición cronológica 70/30 y construcción de las 7 series mensuales | `data/raw/*.xlsx` | `series_mensuales.csv` |
| 3 | `03_serie_total.ipynb` | Análisis completo (incisos 4a–4k) de la serie total | `series_mensuales.csv`, `src/ts_utils.py` | — |
| 4 | `04_series_regiones.ipynb` | Íd. para las 3 regiones de `Región dos` | Íd. | `resumen_regiones.csv` |
| 5 | `05_series_fronteras.ipynb` | Íd. para las 3 fronteras principales | Íd. | `resumen_fronteras.csv` |
| 6 | `06_analisis_comparativo.ipynb` | Comparación entre series y hallazgos para el INGUAT | Los dos resúmenes | — |
