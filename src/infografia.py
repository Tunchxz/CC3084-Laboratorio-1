"""
Cálculos y gráficos de la infografía del Laboratorio 9.

Construye la serie total y las seis series de regiones y fronteras a partir de
la base de migración, calcula las cifras de la infografía (KPI, índice
estacional, caracterización comparativa y métricas de los modelos de la serie
total) y exporta los gráficos.

Uso (desde la raíz del repositorio):
    .venv/bin/python src/infografia.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import font_manager

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "src"))

import ts_utils as tsu

RUTA_DATOS = RAIZ / "data" / "raw" / "Base_Migracion_2009-2026jun.xlsx"
VENTANA_CIERRE_TOTAL = pd.date_range("2020-03-01", "2021-06-01", freq="MS")
ANIOS_EXCLUIDOS_ESTACIONALIDAD = (2020, 2021, 2026)
INTERVALO = 0.95
SEMILLA = 42

NOMBRES = {
    "total": "Total",
    "region_america_central": "América del Centro",
    "region_america_norte": "América del Norte",
    "region_europa": "Europa",
    "frontera_aurora": "01 La Aurora",
    "frontera_valle_nuevo": "07 Valle Nuevo",
    "frontera_san_cristobal": "09 San Cristóbal",
}
REGIONES = {
    "América Del Centro": "region_america_central",
    "América Del Norte": "region_america_norte",
    "Europa": "region_europa",
}
FRONTERAS = {
    "01 La Aurora": "frontera_aurora",
    "07 Valle Nuevo": "frontera_valle_nuevo",
    "09 San Cristóbal": "frontera_san_cristobal",
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def cargar_base(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee la hoja "Datos" y devuelve solo los registros Turista + Excursionista con columna `fecha`."""
    df = pd.read_excel(ruta, sheet_name="Datos")
    for col in ["Año", "Mes cod", "Viajero"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fecha"] = pd.to_datetime(dict(year=df["Año"], month=df["Mes cod"], day=1))
    return df[df["Tipo de Viajero"].isin(["Turista", "Excursionista"])].copy()


def serie_mensual(data: pd.DataFrame, filtro=None) -> pd.Series:
    """Suma `Viajero` por mes; los meses sin registro dentro del cierre de fronteras valen 0."""
    d = data if filtro is None else data[filtro]
    s = d.groupby("fecha")["Viajero"].sum().sort_index().asfreq("MS")
    huecos = s[s.isna()].index.intersection(VENTANA_CIERRE_TOTAL)
    s.loc[huecos] = 0.0
    return s


def construir_series(vis: pd.DataFrame) -> dict[str, pd.Series]:
    """Devuelve las siete series mensuales indexadas por la clave de `NOMBRES`."""
    series = {"total": serie_mensual(vis)}
    for region, clave in REGIONES.items():
        series[clave] = serie_mensual(vis, vis["Región dos"] == region)
    for frontera, clave in FRONTERAS.items():
        series[clave] = serie_mensual(vis, vis["Frontera"] == frontera)
    return series


def particion(s: pd.Series, proporcion: float = 0.70) -> tuple[pd.Series, pd.Series]:
    """Divide la serie cronológicamente en entrenamiento y prueba."""
    corte = int(round(len(s) * proporcion))
    return s.iloc[:corte], s.iloc[corte:]


# ---------------------------------------------------------------------------
# Cifras
# ---------------------------------------------------------------------------

def totales_anuales(s: pd.Series) -> pd.Series:
    """Suma anual de la serie."""
    return s.groupby(s.index.year).sum()


def indice_estacional(s: pd.Series, excluir=ANIOS_EXCLUIDOS_ESTACIONALIDAD) -> pd.Series:
    """Promedio por mes de la razón entre el valor mensual y la media de su año (1 = mes promedio)."""
    d = s.to_frame("v")
    d = d[~d.index.year.isin(excluir)]
    d["razon"] = d["v"] / d.groupby(d.index.year)["v"].transform("mean")
    return d.groupby(d.index.month)["razon"].mean()


def caracterizacion(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Tabla de fuerza de tendencia, fuerza estacional, volatilidad e impacto de la pandemia por serie."""
    filas = [tsu.caracterizar_serie(NOMBRES[k], s, periodo=12) for k, s in series.items()]
    return pd.DataFrame(filas).set_index("serie")


def modelos_total(train: pd.Series, test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta los cinco modelos de la serie total.

    Devuelve la tabla de MAE y RMSE sobre prueba, ordenada por RMSE, y el
    pronóstico de Prophet sobre prueba con columnas `yhat`, `yhat_lower` y
    `yhat_upper` (intervalo `INTERVALO`).
    """
    horizonte = len(test)
    log_train = np.log(train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = tsu.run_auto_arima(log_train, seasonal=True, m=12)
    d = 1
    candidatos = [
        {"order": (1, d, 1), "seasonal_order": (1, 1, 1, 12)},
        {"order": auto.order, "seasonal_order": auto.seasonal_order},
        {"order": (1, d, 0), "seasonal_order": (1, 1, 0, 12)},
        {"order": (0, d, 1), "seasonal_order": (0, 1, 1, 12)},
        {"order": (2, d, 1), "seasonal_order": (1, 1, 1, 12)},
    ]
    tabla_arima, modelos_arima = tsu.fit_arima_grid(log_train, candidatos)
    mejor_arima = modelos_arima[tabla_arima.iloc[0]["modelo"]]

    hw = tsu.fit_holt_winters(train, periodo=12, trend="add", seasonal="mul")
    ses = tsu.fit_simple_exp_smoothing(train)
    np.random.seed(SEMILLA)
    prophet_pred, prophet_full = tsu.fit_prophet_model(train, pasos=horizonte, intervalo=INTERVALO)

    resultados = {
        "ARIMA": {"pred": np.exp(mejor_arima.get_forecast(steps=horizonte).predicted_mean).values},
        "Holt-Winters": {"pred": hw.forecast(horizonte).values},
        "Suavizamiento exp. simple": {"pred": ses.forecast(horizonte).values},
        "Seasonal Naive": {"pred": tsu.seasonal_naive_forecast(train, pasos=horizonte, periodo=12)},
        "Prophet": {"pred": prophet_pred},
    }
    tabla = tsu.tabla_comparativa(resultados, test)[["modelo", "MAE", "RMSE"]]
    pronostico = (prophet_full.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
                  .iloc[-horizonte:].set_axis(test.index))
    return tabla, pronostico


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

CARPETA_FIGURAS = RAIZ / "figures" / "infografia"
CARPETA_FUENTES = RAIZ / "assets" / "fonts"

AZUL = "#1B3A5C"
CELESTE = "#4A90C2"
GRIS = "#6B7280"
NARANJA = "#D9661F"
NARANJA_CLARO = "#F6D2B8"
FONDO = "#F7F8FA"
TEXTO = "#222222"
REJILLA = "#E3E6EA"
RAMPA_AZUL = ("#EAF2F9", "#9CC3E2")

ANCHO_COMPLETO = 9.8
ANCHO_MEDIO = 4.75
ANCHO_MODELOS = 4.2
ANCHO_PRONOSTICO = 4.3
ALTO = 2.9

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
INICIALES_MESES = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

FMT_MILES = mticker.FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:,.0f} mil")


def configurar_estilo() -> None:
    """Registra Open Sans (si está en `assets/fonts`) y fija el estilo común de los gráficos."""
    fuentes = list(CARPETA_FUENTES.glob("*.ttf"))
    for f in fuentes:
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Open Sans" if fuentes else "DejaVu Sans",
        "font.size": 11,
        "text.color": TEXTO,
        "axes.labelcolor": GRIS,
        "axes.edgecolor": GRIS,
        "axes.facecolor": FONDO,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "xtick.color": GRIS,
        "ytick.color": GRIS,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "ytick.left": False,
        "figure.facecolor": FONDO,
        "savefig.facecolor": FONDO,
        "svg.fonttype": "path",
    })


def fecha_es(fecha: pd.Timestamp) -> str:
    """Devuelve la fecha como "mes año" en español."""
    return f"{MESES[fecha.month - 1]} {fecha.year}"


def guardar(fig, nombre: str, carpeta: Path = CARPETA_FIGURAS) -> None:
    """Guarda la figura en SVG (fondo transparente, para Canva) y en PNG a 300 ppp."""
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}.svg", transparent=True, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(carpeta / f"{nombre}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def grafico_historico(total: pd.Series):
    """Línea mensual 2009–2026 con anotaciones de máximo, pandemia, mínimo, reactivación y metodología."""
    fig, ax = plt.subplots(figsize=(ANCHO_COMPLETO, ALTO))
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color=GRIS, alpha=0.12, lw=0)
    ax.axvline(pd.Timestamp("2023-01-01"), color=GRIS, lw=0.8)
    ax.plot(total.index, total.values, color=AZUL, lw=2)

    pico, minimo = total.idxmax(), total.idxmin()
    ax.scatter([pico, minimo], [total[pico], total[minimo]], s=40, color=NARANJA, zorder=3,
               edgecolor=FONDO, linewidth=1.5)
    flecha = dict(arrowstyle="-", color=GRIS, lw=0.8)
    ax.annotate(f"Máximo histórico\n{fecha_es(pico)}: {total[pico]:,.0f}", xy=(pico, total[pico]),
                xytext=(-14, 4), textcoords="offset points", ha="right", va="center", fontsize=11)
    ax.annotate(f"Mínimo: {fecha_es(minimo)}\n{total[minimo]:,.0f} visitantes", xy=(minimo, total[minimo]),
                xytext=(pd.Timestamp("2019-06-01"), total[minimo] + 25000), ha="right", va="bottom",
                fontsize=11, arrowprops=flecha)
    tope = ax.get_ylim()[1] * 1.12
    ax.set_ylim(0, tope)
    ax.text(pd.Timestamp("2020-04-15"), tope * 0.97, "Pandemia y\ncierre de\nfronteras", ha="left",
            va="top", fontsize=11, color=GRIS)
    ax.text(pd.Timestamp("2023-03-01"), tope * 0.03, "Cambio en la\nmetodología de reporte", ha="left",
            va="bottom", fontsize=11, color=GRIS)
    ax.text(pd.Timestamp("2021-04-15"), total.max() * 0.55, "Reactivación\n2022", ha="center", va="bottom",
            fontsize=11)

    ax.set_xlim(total.index[0], total.index[-1] + pd.offsets.MonthBegin(1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(FMT_MILES)
    ax.set_ylabel("Visitantes por mes")
    return fig


def grafico_estacionalidad(indice: pd.Series):
    """Barras de la diferencia porcentual de cada mes frente al mes promedio, con fechas de referencia."""
    desv = (indice - 1) * 100
    colores = [NARANJA if m == 12 else CELESTE for m in desv.index]
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(ANCHO_MEDIO, ALTO))
    ax.bar(x, desv.values, width=0.72, color=colores)
    ax.axhline(0, color=GRIS, lw=0.8)

    def valor(mes, arriba=True):
        v = desv[mes]
        ax.text(mes - 1, v + (1.5 if arriba else -1.5), f"{v:+.0f} %".replace("-", "−"), ha="center",
                va="bottom" if arriba else "top", fontsize=11, fontweight="semibold")

    valor(12)
    valor(2, arriba=False)
    valor(9, arriba=False)
    alto_marzo = max(desv[3], desv[4])
    ax.text(2.5, alto_marzo + 2, "Semana\nSanta", ha="center", va="bottom", fontsize=11, color=GRIS)
    ax.text(6, desv[7] + 2, "Vacaciones\ny bono 14", ha="center", va="bottom", fontsize=11, color=GRIS)
    ax.text(11, desv[12] + 9, "Aguinaldo\ny fiestas", ha="center", va="bottom", fontsize=11, color=GRIS)
    ax.text(5.5, min(desv[2], desv[9]) - 9, "Temporada baja en febrero y septiembre", ha="center",
            va="top", fontsize=11, color=GRIS)

    ax.set_ylim(min(desv) - 16, desv.max() + 22)
    ax.set_xticks(x, INICIALES_MESES)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f} %".replace("-", "−")
                                                       if v else "0"))
    return fig


def grafico_cuadrante(carac: pd.DataFrame):
    """Matriz de las siete series por cuatro métricas; resalta en naranja el valor más alto de cada columna."""
    columnas = [
        ("fuerza_tendencia", "Tendencia de\ncrecimiento", "{:.2f}"),
        ("fuerza_estacional", "Fuerza\nestacional", "{:.2f}"),
        ("volatilidad_residuo", "Volatilidad", "{:.3f}"),
        ("impacto_pandemia_pct", "Caída en\npandemia", "{:.2f} %"),
    ]
    filas = list(carac.index)
    huecos = {1: 0.25, 4: 0.25}
    y, posiciones = 0.0, []
    for i in range(len(filas)):
        y += huecos.get(i, 0.0)
        posiciones.append(y)
        y += 1

    fig, ax = plt.subplots(figsize=(ANCHO_MEDIO, ALTO))
    claro, oscuro = (np.array(mcolors.to_rgb(c)) for c in RAMPA_AZUL)
    for j, (col, titulo, fmt) in enumerate(columnas):
        v = carac[col]
        rango = (v.max() - v.min()) or 1.0
        for i, fila in enumerate(filas):
            maximo = np.isclose(v[fila], v.max())
            color = NARANJA_CLARO if maximo else claro + (oscuro - claro) * (v[fila] - v.min()) / rango
            ax.add_patch(mpatches.Rectangle((j + 0.03, posiciones[i] + 0.04), 0.94, 0.92, color=color, lw=0))
            ax.text(j + 0.5, posiciones[i] + 0.5, fmt.format(v[fila]), ha="center", va="center", fontsize=11,
                    fontweight="bold" if maximo else "normal")
        ax.text(j + 0.5, -0.25, titulo, ha="center", va="bottom", fontsize=11, fontweight="semibold")
    for i, fila in enumerate(filas):
        ax.text(-0.08, posiciones[i] + 0.5, fila, ha="right", va="center", fontsize=11)

    ax.set_xlim(-0.05, len(columnas))
    ax.set_ylim(posiciones[-1] + 1, -1.3)
    ax.axis("off")
    return fig


def grafico_modelos(tabla: pd.DataFrame):
    """Barras horizontales de MAE y RMSE por modelo con la misma escala; el de menor error en naranja."""
    tabla = tabla.sort_values("RMSE").reset_index(drop=True)
    colores = [NARANJA if i == 0 else CELESTE for i in range(len(tabla))]
    tope = tabla[["MAE", "RMSE"]].to_numpy().max() * 1.32
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO_MODELOS, ALTO), sharey=True)
    for ax, metrica, titulo in zip(axes, ["MAE", "RMSE"], ["Error promedio\n(MAE)", "Error cuadrático\n(RMSE)"]):
        ax.barh(tabla["modelo"], tabla[metrica], height=0.62, color=colores)
        for i, v in enumerate(tabla[metrica]):
            ax.text(v + tope * 0.02, i, f"{v / 1000:,.0f} mil", va="center", fontsize=11,
                    fontweight="bold" if i == 0 else "normal")
        ax.set_xlim(0, tope)
        ax.set_title(titulo, fontsize=11, fontweight="semibold", loc="left", color=TEXTO)
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=TEXTO)
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(len(tabla)), [m.replace("Suavizamiento exp. simple", "Suavizamiento\nexp. simple")
                                           for m in tabla["modelo"]])
    fig.subplots_adjust(wspace=0.08)
    return fig


def grafico_pronostico(total: pd.Series, test: pd.Series, pronostico: pd.DataFrame, desde: str = "2019-01-01"):
    """Serie real y pronóstico de Prophet sobre prueba con la banda del intervalo rotulada como Estimación."""
    real = total[desde:]
    inferior = pronostico["yhat_lower"].clip(lower=0)
    fig, ax = plt.subplots(figsize=(ANCHO_PRONOSTICO, ALTO))
    ax.fill_between(pronostico.index, inferior, pronostico["yhat_upper"], color=NARANJA, alpha=0.18, lw=0)
    ax.axvline(test.index[0], color=GRIS, lw=0.8)
    ax.plot(real.index, real.values, color=AZUL, lw=2)
    ax.plot(pronostico.index, pronostico["yhat"], color=NARANJA, lw=2, ls=(0, (4, 2)))

    fin = real.index[-1]
    ax.text(fin + pd.offsets.MonthBegin(2), real.iloc[-6:].mean(), "Real", va="center", fontsize=11,
            fontweight="semibold", color=TEXTO)
    ax.text(fin + pd.offsets.MonthBegin(2), pronostico["yhat"].iloc[-6:].mean(), "Pronóstico", va="center",
            fontsize=11, fontweight="semibold", color=TEXTO)
    ax.text(pd.Timestamp("2024-03-01"), 12000, f"Estimación\n(intervalo {INTERVALO:.0%})".replace("%", " %"),
            ha="center", va="bottom", fontsize=11, color=TEXTO)
    tope = max(real.max(), pronostico["yhat_upper"].max()) * 1.15
    ax.set_ylim(0, tope)
    ax.text(test.index[0] + pd.offsets.MonthBegin(1), tope * 0.98, f"Inicio de la prueba\n{fecha_es(test.index[0])}",
            ha="left", va="top", fontsize=11, color=GRIS)

    ax.set_xlim(real.index[0], fin + pd.offsets.MonthBegin(1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(FMT_MILES)
    ax.set_ylabel("Visitantes por mes")
    return fig


def graficar(datos: dict, carpeta: Path = CARPETA_FIGURAS) -> None:
    """Genera y guarda los cinco gráficos de la infografía a partir del resultado de `calcular`."""
    configurar_estilo()
    guardar(grafico_historico(datos["total"]), "b2_historico", carpeta)
    guardar(grafico_estacionalidad(datos["indice"]), "b3_estacionalidad", carpeta)
    guardar(grafico_cuadrante(datos["caracterizacion"]), "b3_cuadrante", carpeta)
    guardar(grafico_modelos(datos["modelos"]), "b4_modelos", carpeta)
    guardar(grafico_pronostico(datos["total"], datos["test"], datos["pronostico"]), "b4_pronostico", carpeta)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def calcular() -> dict:
    """Calcula todas las cifras y series que usan la infografía y sus gráficos."""
    vis = cargar_base()
    series = construir_series(vis)
    total = series["total"]
    train, test = particion(total)
    modelos, pronostico = modelos_total(train, test)
    regiones = vis[~vis["Región dos"].isin(["Cruceristas", "Cruceros", "0"])]
    return {
        "total": total,
        "train": train,
        "test": test,
        "anual": totales_anuales(total),
        "participacion_region": regiones.groupby("Región dos")["Viajero"].sum() / vis["Viajero"].sum() * 100,
        "participacion_via": vis.groupby("Vía")["Viajero"].sum() / vis["Viajero"].sum() * 100,
        "indice": indice_estacional(total),
        "caracterizacion": caracterizacion(series),
        "modelos": modelos,
        "pronostico": pronostico,
    }


def imprimir_resumen(datos: dict) -> None:
    """Imprime las cifras que se citan en la infografía."""
    total, train, test, anual = datos["total"], datos["train"], datos["test"], datos["anual"]
    pronostico = datos["pronostico"]
    print(f"Meses: {len(total)}  |  entrenamiento {train.index[0]:%Y-%m} a {train.index[-1]:%Y-%m} "
          f"({len(train)})  |  prueba {test.index[0]:%Y-%m} a {test.index[-1]:%Y-%m} ({len(test)})")
    print("\nTotales anuales (millones):")
    print((anual / 1e6).round(2).to_string())
    print(f"\nCaída 2020 vs 2019: {(anual[2020] / anual[2019] - 1) * 100:.1f} %")
    print(f"Máximo mensual: {total.idxmax():%Y-%m} ({total.max():,.0f})")
    print(f"Mínimo mensual: {total.idxmin():%Y-%m} ({total.min():,.0f})")
    print(f"Participación de Centroamérica: {datos['participacion_region']['América Del Centro']:.1f} %")
    print("Participación por vía (%):", datos["participacion_via"].round(1).to_dict())
    print("\nÍndice estacional (años sin 2020, 2021 ni 2026):")
    print(datos["indice"].round(3).to_string())
    print("\nCaracterización comparativa:")
    print(datos["caracterizacion"].round(3).to_string())
    print("\nModelos, serie total (prueba):")
    print(datos["modelos"].round(0).to_string(index=False))
    dentro = (test >= pronostico["yhat_lower"]) & (test <= pronostico["yhat_upper"])
    print(f"\nMeses de prueba dentro del intervalo del {INTERVALO:.0%} de Prophet: {dentro.mean() * 100:.1f} %")


def main() -> None:
    """Calcula las cifras, las imprime y exporta los gráficos a `figures/infografia`."""
    datos = calcular()
    imprimir_resumen(datos)
    graficar(datos)
    print(f"\nGráficos guardados en {CARPETA_FIGURAS}")


if __name__ == "__main__":
    main()
