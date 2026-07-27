# Codebook: ingreso de viajeros internacionales a Guatemala (2009–2026)

Conjunto de datos usado en los notebooks `01_analisis_exploratorio.ipynb` a
`06_analisis_comparativo.ipynb`. Documenta el **ingreso mensual de viajeros internacionales a Guatemala**
por vía de entrada, frontera, país de residencia y tipo de viajero. Los datos provienen del archivo
proporcionado en Canvas (`Base_Migracion_2009-2026jun.xlsx`), que **no se versiona en este repositorio**
por su tamaño y por ser de uso exclusivamente académico: hay que colocarlo en `data/raw/`.

El archivo está en **formato largo**: una fila por combinación de mes, vía, frontera, país/agrupación y
tipo de viajero. No trae filas de totales ni doble conteo, así que cualquier serie de tiempo se obtiene
agregando (`groupby().sum()`) sobre la columna de medida.

## Descripción

Registros de ingreso de viajeros a Guatemala con una observación por combinación de categorías, de
**enero de 2009 a junio de 2026** (**210 meses consecutivos, sin huecos**; **161,036 registros × 13
columnas**). No hay valores nulos ni filas duplicadas exactas, y la frecuencia mensual es regular
(`MS`, inicio de mes) al agregar. La medida (`Viajero`) está en **personas**.

| Variable                | Tipo (Python/pandas) | Descripción                                                     | Valores / notas                                                                                     |
|-------------------------|----------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `Año`                   | int64                | Año de ingreso al país                                          | 2009 – 2026 (18 años) · **2026 cubre solo enero–junio**                                             |
| `Mes cod`               | int64                | Codificación del mes                                            | 1 – 12                                                                                              |
| `Mes`                   | str                  | Nombre abreviado del mes                                        | `Ene` … `Dic` (12 categorías)                                                                       |
| `Vía`                   | str                  | Vía de entrada                                                  | 3 categorías: `Terrestre` (59.1 %), `Aérea` (40.7 %), `Marítima` (0.2 %)                            |
| `Frontera`              | str                  | Frontera o puesto de ingreso                                    | 22 categorías: `01 La Aurora` … `20 Melchor de Mencos`, más `22 Otra frontera` y `Cruceros`         |
| `País`                  | str                  | País de residencia (hasta 2022) o agrupación de mercado (2023+) | 235 valores crudos → **222** al normalizar mayúsculas/espacios · 232 en 2009–2022 vs. **26** en 2023+ |
| `Región`                | str                  | Clasificación usada para reportes nacionales                    | 17 categorías (`AMÉRICA DEL CENTRO`, `EUROPA`, `OTROS PAISES DEL MUNDO`, …)                         |
| `Región dos`            | str                  | Agrupa varias categorías de `Región` en continentes o grandes áreas | 11 categorías · incluye el valor anómalo `'0'` (13 filas, 821 viajeros, todas de 2022)          |
| `Regiones OMT`          | str                  | Subregión de la Organización Mundial del Turismo                | 26 categorías · incluye `SIN ESPECIFICAR` y el valor anómalo `'0x2a'`                               |
| `MCEO`                  | str                  | Mercado o agrupación comercial estratégica                      | 8 categorías (`01 CENTROAMÉRICA` … `06 RESTO DEL MUNDO`, `08 OTROS`, `Cruceros`)                    |
| `Agrupación Residencia` | str                  | Región donde reside el viajero                                  | 33 categorías                                                                                       |
| `Tipo de Viajero`       | str                  | Categoría del viajero                                           | 4 categorías: `Turista`, `Excursionista`, `Viajero`, `Cruceristas` (ver más abajo)                  |
| `Viajero`               | float64              | **Medida:** cantidad de viajeros                                | 0 – 92,336.04 · media 324.70 · mediana 7 · total 52,287,937 · 0.03 % ceros · **31.8 % con decimales** |

Los **decimales** en `Viajero` no son errores de captura, son estimaciones expandidas de encuesta, no
conteos exactos.

## `Tipo de Viajero` y la vista consistente de visitantes

Las cuatro categorías son **independientes** (no se anidan), y su definición está en la hoja `Notas` del
archivo:

| Categoría       | Definición                                                                                        | Filas   | Total viajeros | Años cubiertos |
|-----------------|---------------------------------------------------------------------------------------------------|--------:|---------------:|----------------|
| `Turista`       | Pernocta al menos una noche                                                                       | 117,912 |     37.64 M    | 2009 – 2026    |
| `Excursionista` | Visita sin pernoctar (mismo día)                                                                  |  19,730 |      9.07 M    | 2009 – 2026    |
| `Viajero`       | Cruza la frontera sin calificar como visitante (trabajo fronterizo, tránsito, carga, tripulación) |  23,190 |      4.47 M    | 2009 – 2026    |
| `Cruceristas`   | Pasajeros de crucero                                                                              |     204 |      1.10 M    | **2009 – 2022** |

- **Todas las series de este laboratorio se construyen sobre `Turista + Excursionista`**, la única vista
  comparable en todo el rango. La categoría `Viajero` sufre un quiebre metodológico en 2023 (cae de
  ~1.06 M a ~0.33 M por reclasificación, no por caída real de turismo) y `Cruceristas` desaparece desde
  2023, cuando los cruceros pasaron a medirse por fuente portuaria externa.
- `Viajero` y `Cruceristas` se **exploran** en el EDA pero **se excluyen** de las series modeladas.

## Series de tiempo derivadas

Del archivo crudo se construyen **7 series mensuales** (notebook `02_preparacion_series.ipynb`), todas con
inicio **2009-01**, fin **2026-06**, frecuencia **`MS`** y **210 observaciones sin faltantes**. Se guardan
en `data/processed/series_mensuales.csv` en formato largo (`serie`, `fecha`, `conjunto`, `viajeros`).

| Serie (clave)            | Categoría              | Mínimo | Máximo  | Media   |
|--------------------------|------------------------|-------:|--------:|--------:|
| `total`                  | General                |  9,779 | 449,114 | 222,438 |
| `region_america_central` | Regiones (`Región dos`)|  9,779 | 359,173 | 158,495 |
| `region_america_norte`   | Regiones               |      0 |  98,380 |  43,623 |
| `region_europa`          | Regiones               |      0 |  22,824 |  10,235 |
| `frontera_aurora`        | Fronteras              |    484 | 158,073 |  90,428 |
| `frontera_valle_nuevo`   | Fronteras              |     80 | 111,142 |  48,300 |
| `frontera_san_cristobal` | Fronteras              |     14 |  53,127 |  19,922 |

- Las **dos categorías de análisis** elegidas son **Regiones geográficas** (`Región dos`) y **Fronteras**,
  cada una con sus **3 valores de mayor acumulado en todo el período** (no de un año específico).
- Se descartaron *Países de residencia* (la granularidad cambia de 232 países a 26 agrupaciones en 2023) y
  *Vías de ingreso* (la vía `Marítima` no registra viajeros desde **2017**: 114 meses vacíos).
- Los **5 meses sin registro** en América del Norte y Europa (**abril–agosto 2020**) se imputan a **0**
  únicamente dentro de la ventana de cierre de fronteras (2020-03 → 2021-06), donde el cero es el valor
  real y no un dato faltante. Eso explica el mínimo de 0 en esas dos series.

Los resultados del modelado se persisten en `data/processed/resumen_regiones.csv` y
`resumen_fronteras.csv`, con las columnas `d`, `D`, `descomposición`, `mejor_modelo`, `MAE`, `RMSE` y
`corr_nivel_std`.

## Características relevantes para el modelado

- **Tendencia** creciente y sostenida hasta 2019 (máximo anual de **4.13 M** visitantes), colapso en 2020
  y recuperación posterior. La fuerza de tendencia (Hyndman & Athanasopoulos) va de **0.75** (San
  Cristóbal) a **0.88** (América del Norte).
- **Estacionalidad** anual (período = 12) estable año con año fuera de la pandemia: **pico en diciembre**
  y **valle en febrero**. La fuerza estacional va de **0.30** (San Cristóbal) a **0.73** (Europa).
- **Varianza no estable** en la mayoría de las series: la desviación estándar anual crece con el nivel
  (correlación nivel–desviación de **0.90** en la serie total), por lo que se aplica **transformación
  logarítmica** y descomposición **multiplicativa**. Las excepciones son América del Norte y Europa, que
  contienen ceros y se tratan de forma **aditiva** sin transformar.
- **No estacionaria en media.** El ADF sobre `log(total)` en nivel da **p = 0.1156**; con una
  diferenciación regular baja a **p = 0.0277**, y al añadir la estacional (`D=1`, `s=12`) a **p < 0.001**.
  El orden de diferenciación requerido varía por serie: **d = 0** (América del Norte), **d = 1** (total,
  América del Centro, La Aurora, Valle Nuevo) y **d = 2** (Europa, San Cristóbal), con **D = 1** en todas.
- **Ruptura estructural por pandemia** que domina el final del entrenamiento: el mínimo mensual
  (**mayo 2020, 9,779 viajeros**) equivale a **~2.2 %** del pico previo (diciembre 2019, 449,114). La
  caída máxima frente al nivel de 2019 va de **96.4 %** (América del Centro) a **100 %** en las series que
  llegaron a cero.
- **Atípicos:** el criterio IQR marca 16.4 % de las filas, que concentran 96.7 % del volumen, por lo que
  **no se eliminan**: son estacionalidad y escala legítimas, no errores.

## Partición

Partición **cronológica 70/30**. Se aplica sobre los **210 meses**.

| Conjunto      | Rango             | Meses | %    |
|---------------|-------------------|------:|-----:|
| Entrenamiento | 2009-01 → 2021-03 |   147 | 70 % |
| Prueba        | 2021-04 → 2026-06 |    63 | 30 % |

El corte cae justo antes de la recuperación pospandemia, así que el conjunto de prueba (63 meses) incluye
tanto esa recuperación como el quiebre metodológico de 2023, un horizonte exigente para evaluar
exactitud.

## Fuente

Ingreso de viajeros internacionales a Guatemala, integrado de **tres tramos**:

| Tramo             | Origen                                                        |
|-------------------|---------------------------------------------------------------|
| 2009 – 2020       | Respaldos históricos                                          |
| 2021 – 2022       | Entrega del Instituto Guatemalteco de Migración (IGM)         |
| 2023 – junio 2026 | Sistema depurado de conteos del INGUAT (metodología de boletín) |

Procesamiento ya aplicado en el archivo: agrupación de registros transaccionales para 2021–2022
(totales conservados exactos); para 2023–2026, unión de conteos mensuales con catálogos (fronteras por
código [1–20], vía derivada del código, tipos confirmados y agrupaciones de país), descarte de 2 filas sin
agrupación y etiquetado del código de frontera 22 como `Otra frontera`.

> **Uso exclusivamente académico.** Los datos **no corresponden a cifras oficiales** del INGUAT ni del
> Instituto Guatemalteco de Migración, y ninguna cifra de este repositorio debe citarse como estadística
> oficial.
