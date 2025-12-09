import numpy as np
import pandas as pd

from pathlib import Path

from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression

from .models import (
    quantiles,
    get_model,
    TrainingWindowType,
)


def run_model(
    iso: str,
    model_name: str,
    training_window_type: TrainingWindowType = TrainingWindowType.EXPANDING,
    transform: callable = lambda x: np.log1p(x),
    output_dir: str = "outputs",
):
    # Sanitise inputs
    if not isinstance(iso, str) or not isinstance(model_name, str):
        raise ValueError("ISO and model must be strings")
    iso = iso.upper()
    model_name = model_name.lower()

    # Parameters
    start_time = "2018-01"
    end_time = "2099-01"
    max_horizon = 6
    num_samples = 1000

    # Load composite dataset (target & covariates)
    dirstem = Path(__file__).parent.parent.parent
    df = pd.read_csv(dirstem / "data" / f"Covariates_data_{iso}.csv")
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M")
    df["time"] = df["time"].dt.to_timestamp()

    # Select model
    label, model = get_model(
        model_name=model_name,
        training_window_type=training_window_type,
        max_horizon=max_horizon,
    )

    # Transform target variable
    target_col = "Cases"
    df[target_col] = transform(df[target_col])

    # Construct covariates
    df["Lag_Cases_1"] = transform(df["Cases"]).shift(1)
    df["spe03_2"] = df["spe03"].shift(2)
    df["spa03_2"] = df["spa03"].shift(2)
    df["ssta_2"] = df["ssta"].shift(2)

    covar_list = ["Lag_Cases_1", "spe03_2", "spa03_2", "ssta_2"]
    df = df[[target_col, *covar_list, "time"]]  # subset columns
    df = df[2:]  # cut lag-induced NaNs

    # Build DARTS TimeSeries
    series = TimeSeries.from_dataframe(
        df,
        time_col="time",
        value_cols=["Cases", *covar_list],
    )
    series = series.astype(np.float32)

    forecast_dfs = []
    times = sorted(df["time"][(df["time"] >= start_time) & (df["time"] <= end_time)])
    for time in times:
        print(f"Fitting {label} for time {time}")

        ts = series.drop_after(time)

        # Limit history
        # target=target[-24:]
        # cov=cov[-24:]

        adjust_time = 0  # expanding window adjustment
        if training_window_type == TrainingWindowType.EXPANDING:
            output_chunk_length = max_horizon
            input_chunk_length = sum(series.time_index < time) - output_chunk_length
            model.reform(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                likelihood=QuantileRegression(quantiles=quantiles),
            )
            adjust_time = output_chunk_length

        model.fit(
            ts[target_col],
            past_covariates=ts[covar_list],
            # future_covariates=series[covar_list],
        )
        forecast = model.predict(max_horizon, num_samples=num_samples)

        # Reframe forecast as pandas DataFrame
        forecast_df = forecast.quantile(quantiles).to_dataframe()
        forecast_df["horizon"] = list(range(1, len(forecast_df) + 1))
        forecast_df = forecast_df.reset_index().melt(
            id_vars=["time", "horizon"],
            var_name="quantile",
            value_name="Cases",
        )

        # Append predictions at timepoint
        forecast_dfs.append(
            pd.DataFrame(
                {
                    "model": label,
                    "time": forecast_df["time"] + pd.DateOffset(months=-adjust_time),
                    "horizon": forecast_df["horizon"],
                    "quantile": forecast_df["quantile"],
                    "Cases": forecast_df["Cases"],
                }
            )
        )

    # Combine timepoints into a single DataFrame
    forecast_df = pd.concat(forecast_dfs)

    # Save forecast
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(
        output_path / f"forecast_{label}_{training_window_type.value}_{iso}.csv",
    )
