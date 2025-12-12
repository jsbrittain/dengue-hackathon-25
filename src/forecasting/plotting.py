import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import r2_score

from .wis import compute_wis
from .models import (
    TrainingWindowType,
)


def plot_historical_forecast(
    label: str,
    training_window_type: TrainingWindowType,
    iso: str,
    transform: callable = lambda x: np.log1p(x),
    itransform: callable = lambda y: np.expm1(y),
    output_dir: str = "outputs",
):
    # Load target series
    dirstem = Path(__file__).parent.parent.parent
    cases_df = pd.read_csv(dirstem / "data" / f"Covariates_data_{iso}.csv")
    cases_df = cases_df[["time", "Cases"]]
    cases_df["time"] = pd.to_datetime(cases_df["time"])
    cases_df["Cases"] = transform(cases_df["Cases"])

    forecast_df = pd.read_csv(
        dirstem
        / output_dir
        / f"forecast_{label}_{training_window_type.value}_{iso}.csv"
    )
    forecast_df["time"] = pd.to_datetime(forecast_df["time"])

    # Plot forecasts
    horizons = [1, 3, 6]
    plt.figure()
    for ix, horizon in enumerate(horizons):
        plt.subplot(len(horizons), 1, ix + 1)
        df = forecast_df[forecast_df["horizon"] == horizon]
        df_wide = df.pivot(
            index="time",
            columns="quantile",
            values="Cases",
        ).sort_index()
        # 90% interval
        plt.fill_between(
            df_wide.index,
            itransform(df_wide["Cases_q0.050"]),
            itransform(df_wide["Cases_q0.950"]),
            color="steelblue",
            alpha=0.2,
            label="90% Prediction Interval",
        )
        # 50% interval
        plt.fill_between(
            df_wide.index,
            itransform(df_wide["Cases_q0.250"]),
            itransform(df_wide["Cases_q0.750"]),
            color="steelblue",
            alpha=0.5,
            label="50% Prediction Interval",
        )
        # Case data
        plt.plot(
            cases_df["time"],
            itransform(cases_df["Cases"]),
            label="Original Series",
        )
        # Median
        plt.plot(
            df_wide.index,
            itransform(df_wide["Cases_q0.500"]),
            label=f"Forecast Horizon {horizon} - {label}",
        )
        plt.xlim([cases_df["time"].min(), forecast_df["time"].max()])
    plt.show()


# def r2(y_true, y_pred):
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     return 1 - (ss_res / ss_tot)


def wis(y_true, y_pred_intervals, alphas=[0.5, 0.1]):
    wis_score = 0.0
    for alpha in alphas:
        lower = y_pred_intervals[f"lower_{alpha}"]
        upper = y_pred_intervals[f"upper_{alpha}"]
        interval_width = upper - lower
        penalty = (2 / alpha) * (
            (lower - y_true).clip(lower=0) + (y_true - upper).clip(lower=0)
        )
        wis_score += np.mean(interval_width + penalty)
    wis_score /= len(alphas)
    return wis_score


def plot_historical_forecasts(
    label: str,
    training_window_type: TrainingWindowType,
    isos: list[str],
    transform: callable = lambda x: np.log1p(x),
    itransform: callable = lambda y: np.expm1(y),
    folder: str = "outputs",
    filename: str = None,
    include_wis: bool = False,
):
    dirstem = Path(__file__).parent.parent.parent
    forecasts_df = pd.read_csv(dirstem / folder / filename)
    forecasts_df["time"] = pd.to_datetime(forecasts_df["time"])

    # forecasts_df = forecasts_df[
    #     (forecasts_df['time'] >= '2021-01-01') & (forecasts_df['time'] < '2022-01-01')
    # ]

    plt.figure()
    for iso_idx, iso in enumerate(isos):
        # Load target series
        cases_df = pd.read_csv(dirstem / "data" / f"Covariates_data_{iso}.csv")
        cases_df = cases_df[["time", "Cases"]]
        cases_df["time"] = pd.to_datetime(cases_df["time"])
        cases_df["Cases"] = transform(cases_df["Cases"])

        forecast_df = forecasts_df[forecasts_df["region"] == iso]

        # Plot forecasts
        horizons = [1, 3, 6]
        for ix, horizon in enumerate(horizons):
            ax1 = plt.subplot(len(horizons), len(isos), ix * len(isos) + iso_idx + 1)
            if include_wis:
                ax2 = ax1.twinx()
            df = forecast_df[forecast_df["horizon"] == horizon]

            # Metrics

            cases_df["region"] = iso
            wis_df = compute_wis(df_pred=df, df_true=cases_df, itransform=itransform)

            prediction = df[df["quantile"] == 0.500][["time", "Cases"]]
            merged = prediction.merge(
                cases_df, on="time", how="inner", suffixes=("_pred", "_true")
            ).sort_values("time")
            r2 = r2_score(
                itransform(merged["Cases_true"].values),
                itransform(merged["Cases_pred"].values),
            )

            if True:
                # 90% interval
                ax1.fill_between(
                    df[df["quantile"] == 0.500]["time"],
                    itransform(df[df["quantile"] == 0.050]["Cases"]),
                    itransform(df[df["quantile"] == 0.950]["Cases"]),
                    color="steelblue",
                    alpha=0.2,
                    label="90% Prediction Interval",
                )
                # 50% interval
                ax1.fill_between(
                    df[df["quantile"] == 0.500]["time"],
                    itransform(df[df["quantile"] == 0.250]["Cases"]),
                    itransform(df[df["quantile"] == 0.750]["Cases"]),
                    color="steelblue",
                    alpha=0.5,
                    label="50% Prediction Interval",
                )
                # Case data
                ax1.plot(
                    cases_df["time"],
                    itransform(cases_df["Cases"]),
                    label="Original Series",
                )
                # Median
                ax1.plot(
                    df[df["quantile"] == 0.500]["time"],
                    itransform(df[df["quantile"] == 0.500]["Cases"]),
                    label=f"Forecast Horizon {horizon} - {label}",
                )

            if include_wis:
                ax2.plot(
                    wis_df["time"],
                    wis_df["WIS"],
                    color="red",
                    label="WIS",
                )

            plt.xlim([cases_df["time"].min(), forecast_df["time"].max()])

            # Title
            plt.title(f"R^2 = {r2:.2f} WIS = {wis_df['WIS'].mean():.2f}")
            print(f"{iso}\t{horizon}\tR2\t{r2:.2f}")
            print(f"{iso}\t{horizon}\tWIS\t{wis_df['WIS'].mean():.2f}")
    plt.show()
