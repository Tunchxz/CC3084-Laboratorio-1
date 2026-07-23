"""
Utilidades compartidas para el análisis de series de tiempo.
Centraliza el pipeline que se repite para cada serie: 
    - descripción, descomposición, pruebas de estacionariedad, ACF/PACF,
      modelos ARIMA/benchmarks y evaluación
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython.display import display

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

MILES = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")


# ---------------------------------------------------------------------------
# Descripción básica y visualización
# ---------------------------------------------------------------------------

def series_summary(s: pd.Series, nombre: str = "serie") -> None:
    """Imprime inicio, fin, frecuencia y número de observaciones (inciso 4a)."""
    freq = pd.infer_freq(s.index) or "MS (mensual, inferida por construcción)"
    print(f"Serie: {nombre}")
    print(f"  Inicio:        {s.index.min():%Y-%m}")
    print(f"  Fin:           {s.index.max():%Y-%m}")
    print(f"  Frecuencia:    {freq}")
    print(f"  Observaciones: {len(s)}")
    print(f"  Faltantes:     {s.isna().sum()}")


def plot_series(s: pd.Series, titulo: str, color="C0", ax=None, resaltar_pandemia=True):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, lw=1.4, color=color)
    if resaltar_pandemia:
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color="grey", alpha=0.12)
    ax.set(title=titulo, xlabel="Mes", ylabel="Viajeros")
    ax.yaxis.set_major_formatter(MILES)
    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Descomposición y estacionariedad (incisos 4c, 4d, 4e)
# ---------------------------------------------------------------------------

def decompose_series(s: pd.Series, modelo: str = "additive", periodo: int = 12, titulo: str = ""):
    """Descompone la serie (tendencia/estacionalidad/residuo) y grafica."""
    resultado = seasonal_decompose(s, model=modelo, period=periodo, extrapolate_trend="freq")
    fig = resultado.plot()
    fig.set_size_inches(9, 7)
    fig.suptitle(titulo or "Descomposición de la serie", y=1.02, fontweight="bold")
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(MILES)
    plt.tight_layout()
    plt.show()
    return resultado


def variance_stability_check(s: pd.Series, periodo: int = 12) -> pd.DataFrame:
    """Compara la desviación estándar por año para evaluar estacionariedad en varianza."""
    df = s.to_frame("valor")
    df["anio"] = df.index.year
    resumen = df.groupby("anio")["valor"].agg(["mean", "std"])
    resumen["cv"] = resumen["std"] / resumen["mean"]
    return resumen


def adf_report(s: pd.Series, etiqueta: str = "serie", alpha: float = 0.05) -> dict:
    """Ejecuta la prueba ADF e imprime una interpretación (inciso 4e.ii)."""
    stat, pvalue, nlags, nobs, crit, *_ = adfuller(s.dropna(), autolag="AIC")
    es_estacionaria = pvalue < alpha
    print(f"ADF ({etiqueta}): estadístico={stat:.4f}  p-valor={pvalue:.4f}  lags usados={nlags}  n obs={nobs}")
    print("  Valores críticos:", {k: round(v, 3) for k, v in crit.items()})
    veredicto = "ESTACIONARIA" if es_estacionaria else "NO estacionaria"
    print(f"  Conclusión (alpha={alpha}): la serie es {veredicto} en media.")
    return {"stat": stat, "pvalue": pvalue, "nlags": nlags, "crit": crit, "estacionaria": es_estacionaria}


def plot_acf_pacf(s: pd.Series, lags: int = 36, titulo: str = ""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    plot_acf(s.dropna(), lags=lags, ax=ax[0])
    plot_pacf(s.dropna(), lags=lags, ax=ax[1], method="ywm")
    ax[0].set_title(f"ACF {titulo}")
    ax[1].set_title(f"PACF {titulo}")
    plt.tight_layout()
    plt.show()


def find_d(s: pd.Series, max_d: int = 2, alpha: float = 0.05) -> tuple[int, pd.Series]:
    """Diferencia sucesivamente hasta lograr estacionariedad en media (ADF), o hasta max_d."""
    d = 0
    serie_d = s.copy()
    while d < max_d:
        pvalue = adfuller(serie_d.dropna(), autolag="AIC")[1]
        print(f"  d={d}: ADF p-valor={pvalue:.4f}")
        if pvalue < alpha:
            break
        serie_d = serie_d.diff().dropna()
        d += 1
    else:
        pvalue = adfuller(serie_d.dropna(), autolag="AIC")[1]
        print(f"  d={d}: ADF p-valor={pvalue:.4f}")
    print(f"  -> Diferenciaciones necesarias: d={d}")
    return d, serie_d


# ---------------------------------------------------------------------------
# Modelos ARIMA (incisos 4f, 4g)
# ---------------------------------------------------------------------------

def fit_arima_grid(train: pd.Series, candidatos: list) -> pd.DataFrame:
    """Ajusta varios ARIMA(p,d,q)(P,D,Q,m) y compara AIC/BIC/residuos (inciso 4g).

    `candidatos` es una lista de dicts {"order": (p,d,q), "seasonal_order": (P,D,Q,m)}.
    `seasonal_order` es opcional (por defecto sin componente estacional).
    """
    filas = []
    modelos = {}
    for spec in candidatos:
        order = spec["order"]
        seasonal_order = spec.get("seasonal_order", (0, 0, 0, 0))
        clave = f"{order}x{seasonal_order}" if seasonal_order != (0, 0, 0, 0) else str(order)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(train, order=order, seasonal_order=seasonal_order).fit()
            resid = res.resid
            filas.append({
                "modelo": clave, "order": order, "seasonal_order": seasonal_order,
                "AIC": res.aic, "BIC": res.bic,
                "media_residuo": resid.mean(), "std_residuo": resid.std(),
            })
            modelos[clave] = res
        except Exception:
            filas.append({"modelo": clave, "order": order, "seasonal_order": seasonal_order,
                          "AIC": np.nan, "BIC": np.nan,
                          "media_residuo": np.nan, "std_residuo": np.nan})
    tabla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    return tabla, modelos


def caracterizar_serie(nombre: str, s: pd.Series, periodo: int = 12) -> dict:
    """
    Caracteriza una serie completa (train+test) para el análisis comparativo
    (ejercicio 5): fuerza de tendencia y estacionalidad (medidas de Hyndman &
    Athanasopoulos, basadas en la descomposición aditiva), volatilidad del
    residuo, e impacto porcentual de la pandemia (nivel 2019 vs. mínimo 2020).
    """
    desc = seasonal_decompose(s, model="additive", period=periodo, extrapolate_trend="freq")
    var_resid = desc.resid.var()
    fuerza_estacional = max(0.0, 1 - var_resid / (desc.seasonal + desc.resid).var())
    fuerza_tendencia = max(0.0, 1 - var_resid / (desc.trend + desc.resid).var())
    volatilidad = desc.resid.std() / s.mean()

    base_2019 = s[s.index.year == 2019].mean()
    minimo_2020 = s[s.index.year == 2020].min()
    impacto_pandemia_pct = (base_2019 - minimo_2020) / base_2019 * 100

    return {
        "serie": nombre,
        "fuerza_tendencia": fuerza_tendencia,
        "fuerza_estacional": fuerza_estacional,
        "volatilidad_residuo": volatilidad,
        "impacto_pandemia_pct": impacto_pandemia_pct,
    }


def diagnostico_residuos(res, nombre: str = "modelo", lags=(12, 24)):
    """Grafica los residuos y aplica Ljung-Box para ver si se comportan como ruido blanco."""
    resid = res.resid
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    ax[0].plot(resid, lw=1); ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_title(f"Residuos — {nombre}")
    plot_acf(resid.dropna(), lags=24, ax=ax[1]); ax[1].set_title("ACF residuos")
    ax[2].hist(resid.dropna(), bins=25, color="C0"); ax[2].set_title("Distribución residuos")
    plt.tight_layout(); plt.show()

    lb = acorr_ljungbox(resid.dropna(), lags=list(lags), return_df=True)
    print(f"Ljung-Box ({nombre}):")
    display(lb)
    ok = (lb["lb_pvalue"] > 0.05).all()
    print("  -> Residuos compatibles con ruido blanco (no autocorrelación)." if ok
          else "  -> Aún queda autocorrelación en los residuos (p-valor < 0.05 en algún lag).")
    return lb


def run_auto_arima(train: pd.Series, seasonal: bool = True, m: int = 12):
    """Sugerencia automática de (p,d,q)(P,D,Q,m) vía pmdarima, para contrastar
    con la elección manual basada en ACF/PACF (inciso 4f)."""
    import pmdarima as pm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = pm.auto_arima(
            train, seasonal=seasonal, m=m,
            stepwise=True, suppress_warnings=True, trace=False,
        )
    return modelo


# ---------------------------------------------------------------------------
# Benchmarks: Holt-Winters, suavizamiento exponencial, seasonal naive (inciso 4h)
# ---------------------------------------------------------------------------

def fit_holt_winters(train: pd.Series, periodo: int = 12, trend="add", seasonal="add"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            train, trend=trend, seasonal=seasonal, seasonal_periods=periodo,
            initialization_method="estimated",
        ).fit()
    return modelo


def fit_simple_exp_smoothing(train: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    return modelo


def seasonal_naive_forecast(train: pd.Series, pasos: int, periodo: int = 12) -> np.ndarray:
    """y_hat[t] = y[t - periodo] usando el último ciclo estacional observado."""
    ultimo_ciclo = train.iloc[-periodo:].values
    reps = int(np.ceil(pasos / periodo))
    return np.tile(ultimo_ciclo, reps)[:pasos]


def fit_prophet_model(train: pd.Series, pasos: int):
    """Ajusta Prophet; retorna (forecast_values, componentes) o (None, None) si no está disponible."""
    try:
        from prophet import Prophet
    except ImportError:
        print("  Prophet no está instalado en este entorno; se omite este benchmark.")
        return None, None

    df_train = pd.DataFrame({"ds": train.index, "y": train.values})
    modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(df_train)
    futuro = modelo.make_future_dataframe(periods=pasos, freq="MS")
    pronostico = modelo.predict(futuro)
    valores = pronostico.set_index("ds")["yhat"].iloc[-pasos:].values
    return valores, pronostico


# ---------------------------------------------------------------------------
# Evaluación y comparación (incisos 4i, 4j, 4k)
# ---------------------------------------------------------------------------

def evaluar(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    return {"MAE": mae, "RMSE": rmse}


def tabla_comparativa(resultados: dict, y_test: pd.Series) -> pd.DataFrame:
    """
    resultados: dict {nombre_modelo: {"pred": array_like, "AIC": float|None, "BIC": float|None}}
    """
    filas = []
    for nombre, info in resultados.items():
        pred = np.asarray(info["pred"])
        n = min(len(pred), len(y_test))
        metricas = evaluar(y_test.values[:n], pred[:n])
        filas.append({
            "modelo": nombre, "MAE": metricas["MAE"], "RMSE": metricas["RMSE"],
            "AIC": info.get("AIC"), "BIC": info.get("BIC"),
        })
    return pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)


def plot_forecast_comparison(train: pd.Series, test: pd.Series, forecasts: dict, titulo: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train.index, train.values, color="grey", lw=1.1, label="Entrenamiento")
    ax.plot(test.index, test.values, color="black", lw=1.6, label="Real (prueba)")
    for nombre, pred in forecasts.items():
        n = min(len(pred), len(test))
        ax.plot(test.index[:n], np.asarray(pred)[:n], lw=1.3, ls="--", label=nombre)
    ax.set(title=titulo, xlabel="Mes", ylabel="Viajeros")
    ax.yaxis.set_major_formatter(MILES)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Pipeline completo por serie (reutilizado en los notebooks 04 y 05, que
# analizan varias series de la misma categoría sin repetir todo el código
# de 03_serie_total.ipynb)
# ---------------------------------------------------------------------------

def analizar_serie_completa(nombre: str, train: pd.Series, test: pd.Series, periodo: int = 12) -> dict:
    """
    Ejecuta el pipeline completo del ejercicio 4 (incisos a-k) para una serie:
    resumen, gráfico, descomposición, estacionariedad, ACF/PACF, selección de
    ARIMA, benchmarks (Holt-Winters, suav. exp. simple, seasonal naive, Prophet),
    pronóstico y comparación de métricas. Imprime y grafica cada paso, y
    devuelve un dict con los resultados clave para el resumen comparativo
    (notebook 06).
    """
    horizonte = len(test)
    print("=" * 78)
    print(f"SERIE: {nombre}")
    print("=" * 78)

    # 4a
    series_summary(train.combine_first(test) if False else pd.concat([train, test]), nombre)

    # 4b
    plot_series(pd.concat([train, test]), f"{nombre}: serie mensual completa")
    plt.axvline(train.index[-1], color="red", ls="--", lw=1)
    plt.show()

    # 4c
    resumen_var = variance_stability_check(train)
    anios_normales = resumen_var.drop(index=[2020, 2021], errors="ignore")
    corr = anios_normales["mean"].corr(anios_normales["std"])
    print(f"Correlación nivel medio anual vs. desviación estándar anual (sin 2020-2021): {corr:.2f}")
    tiene_ceros = (train <= 0).any()
    modelo_decomp = "multiplicative" if (corr > 0.3 and not tiene_ceros) else "additive"
    if tiene_ceros and corr > 0.3:
        print("  Nota: la serie tiene meses en 0 (cierre total de fronteras en pandemia) -> "
              "la descomposición multiplicativa y el logaritmo no están definidos ahí; se usa aditiva.")
    print(f"-> Se usa descomposición {modelo_decomp} "
          f"({'la amplitud estacional crece con el nivel' if modelo_decomp=='multiplicative' else 'la amplitud estacional es aprox. constante o hay ceros en la serie'}).")
    _ = decompose_series(train, modelo=modelo_decomp, periodo=periodo, titulo=f"Descomposición -- {nombre}")

    # 4d: transformar si es multiplicativa
    usa_log = modelo_decomp == "multiplicative"
    y_train = np.log(train) if usa_log else train.copy()
    print(f"Transformación aplicada: {'logaritmo (np.log)' if usa_log else 'ninguna'}.")

    # 4e: ADF + diferenciación regular
    print("\nADF sobre la serie (transformada) en nivel:")
    adf_report(y_train, nombre)
    print("\nBúsqueda de 'd' (diferenciación regular):")
    d, y_train_d = find_d(y_train, max_d=2)
    plot_acf_pacf(y_train_d, lags=36, titulo=f"-- {nombre} (d={d})")

    # Diferenciación estacional adicional
    y_train_sd = y_train_d.diff(periodo).dropna()
    print("\nADF tras diferenciación regular + estacional (D=1):")
    adf_report(y_train_sd, f"{nombre}, d+D=1({periodo})")
    D = 1

    # 4f: auto_arima + elección manual
    auto_model = run_auto_arima(y_train, seasonal=True, m=periodo)
    manual_order = (1, d, 1)
    manual_seasonal = (1, D, 1, periodo)
    print(f"\nElección manual (ACF/PACF): order={manual_order}  seasonal_order={manual_seasonal}")
    print(f"auto_arima:                 order={auto_model.order}  seasonal_order={auto_model.seasonal_order}")

    # 4g: grid ARIMA
    candidatos = [
        {"order": manual_order, "seasonal_order": manual_seasonal},
        {"order": auto_model.order, "seasonal_order": auto_model.seasonal_order},
        {"order": (1, d, 0), "seasonal_order": (1, D, 0, periodo)},
        {"order": (0, d, 1), "seasonal_order": (0, D, 1, periodo)},
    ]
    tabla_arima, modelos_arima = fit_arima_grid(y_train, candidatos)
    display(tabla_arima)
    mejor_clave = tabla_arima.iloc[0]["modelo"]
    mejor_arima = modelos_arima[mejor_clave]
    print(f"Mejor ARIMA por AIC: {mejor_clave}")
    diagnostico_residuos(mejor_arima, nombre=f"ARIMA {mejor_clave} ({nombre})")

    # 4h: benchmarks
    hw = fit_holt_winters(train, periodo=periodo, trend="add", seasonal=("mul" if usa_log else "add"))
    ses = fit_simple_exp_smoothing(train)
    naive_pred = seasonal_naive_forecast(train, pasos=horizonte, periodo=periodo)
    prophet_pred, _ = fit_prophet_model(train, pasos=horizonte)

    # 4i: pronósticos (revirtiendo transformación si aplica)
    fc_log = mejor_arima.get_forecast(steps=horizonte).predicted_mean
    pred_arima = np.exp(fc_log) if usa_log else fc_log
    pred_hw = hw.forecast(horizonte)
    pred_ses = ses.forecast(horizonte)

    forecasts = {
        f"ARIMA {mejor_clave}": np.asarray(pred_arima),
        "Holt-Winters": np.asarray(pred_hw),
        "Suav. exp. simple": np.asarray(pred_ses),
        "Seasonal naive": naive_pred,
    }
    if prophet_pred is not None:
        forecasts["Prophet"] = prophet_pred
    plot_forecast_comparison(train, test, forecasts, f"{nombre} -- pronóstico vs. real")

    # 4j, 4k: comparación y selección
    resultados = {
        f"ARIMA {mejor_clave}": {"pred": forecasts[f"ARIMA {mejor_clave}"], "AIC": mejor_arima.aic, "BIC": mejor_arima.bic},
        "Holt-Winters": {"pred": forecasts["Holt-Winters"], "AIC": hw.aic, "BIC": hw.bic},
        "Suav. exp. simple": {"pred": forecasts["Suav. exp. simple"], "AIC": ses.aic, "BIC": ses.bic},
        "Seasonal naive": {"pred": forecasts["Seasonal naive"], "AIC": None, "BIC": None},
    }
    if prophet_pred is not None:
        resultados["Prophet"] = {"pred": prophet_pred, "AIC": None, "BIC": None}

    tabla_final = tabla_comparativa(resultados, test)
    display(tabla_final.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "AIC": "{:,.1f}", "BIC": "{:,.1f}"}))
    ganador = tabla_final.iloc[0]
    print(f"Mejor modelo para {nombre}: {ganador['modelo']}  "
          f"(MAE={ganador['MAE']:,.0f}, RMSE={ganador['RMSE']:,.0f})")

    return {
        "nombre": nombre, "d": d, "D": D, "modelo_decomp": modelo_decomp,
        "mejor_arima_orden": mejor_clave, "auto_arima_orden": (auto_model.order, auto_model.seasonal_order),
        "tabla_metricas": tabla_final, "mejor_modelo": ganador["modelo"],
        "mejor_mae": ganador["MAE"], "mejor_rmse": ganador["RMSE"],
        "train": train, "test": test, "corr_nivel_std": corr,
    }
