import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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
        # 90% interval
        plt.fill_between(
            df[df["quantile"] == "Cases_q0.500"]["time"],
            itransform(df[df["quantile"] == "Cases_q0.050"]["Cases"]),
            itransform(df[df["quantile"] == "Cases_q0.950"]["Cases"]),
            color="steelblue",
            alpha=0.2,
            label="90% Prediction Interval",
        )
        # 50% interval
        plt.fill_between(
            df[df["quantile"] == "Cases_q0.500"]["time"],
            itransform(df[df["quantile"] == "Cases_q0.250"]["Cases"]),
            itransform(df[df["quantile"] == "Cases_q0.750"]["Cases"]),
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
            df[df["quantile"] == "Cases_q0.500"]["time"],
            itransform(df[df["quantile"] == "Cases_q0.500"]["Cases"]),
            label=f"Forecast Horizon {horizon} - {label}",
        )
    plt.show()
