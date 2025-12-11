import numpy as np
import pandas as pd

from typing import List
from pathlib import Path

from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression

from .models import (
    quantiles,
    get_model,
    TrainingWindowType,
)


def run_model(
    isos: List[str],
    model_name: str,
    training_window_type: TrainingWindowType = TrainingWindowType.EXPANDING,
    transform: callable = lambda x: np.log1p(x),
    construct_covariates: callable = lambda df: df,
    output_dir: str = "outputs",
):
    # Sanitise inputs
    iso_name = "composite" if len(isos) > 1 else isos[0].upper()
    model_name = model_name.lower()

    # Parameters
    max_horizon = 6
    num_samples = 1000
    min_history = 24

    # Load composite dataset (target & covariates)
    dirstem = Path(__file__).parent.parent.parent
    df = []
    target_cols = []
    for iso in isos:
        df1 = pd.read_csv(dirstem / "data" / f"Covariates_data_{iso.upper()}.csv")
        if len(isos) > 1:
            if len(df) == 0:
                df = df1

            col = f"Cases_{iso.upper()}"
            df[col] = df1["Cases"]
            target_cols.append(col)

            df[f"pop_{iso.upper()}"] = df1["pop_count"]
        else:
            df = df1
            target_cols = ["Cases"]
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M")
    df["time"] = df["time"].dt.to_timestamp()

    # Select model
    label, model = get_model(
        model_name=model_name,
        training_window_type=training_window_type,
        max_horizon=max_horizon,
    )

    # Transform target variable
    df[target_cols] = transform(df[target_cols])

    # Construct covariates
    covar = construct_covariates(df)
    if covar.shape[1] > 0:
        df = df.join(covar)
    covar_list = covar.columns.tolist()
    df = df[["time", *target_cols, *covar_list]]  # subset columns
    start_idx = covar.dropna().index[0]
    df = df[start_idx:].reset_index(drop=True)  # drop lag-induced NaNs
    times = sorted(df["time"].unique())

    # Build DARTS TimeSeries
    series_target = TimeSeries.from_dataframe(
        df,
        time_col="time",
        value_cols=target_cols,
    ).astype(np.float32)
    if covar_list:
        series_covs = TimeSeries.from_dataframe(
            df,
            time_col="time",
            value_cols=covar_list,
        ).astype(np.float32)

    forecast_dfs = []
    for idx in range(len(series_target)):
        time = series_target.time_index[idx]

        print(f"Fitting {label} for time {time}")
        if time <= times[min_history + max_horizon - 1]:
            continue

        origin_idx = np.where(series_target.time_index == time)[0][0]
        ts_target = series_target[: origin_idx + 1]  # inclusive slice
        # ts_target = series_target.drop_after(time)
        if covar_list:
            # ts_covars = series_covs.drop_after(time)
            ts_covars = series_covs[: origin_idx + 1]  # inclusive slice

        print("loop time:            ", time)
        print("ts_target.last_index: ", ts_target.time_index[-1])
        assert ts_target.time_index[-1] == time

        adjust_time = 0  # expanding window adjustment
        horizon = max_horizon  # default horizon
        output_chunk_shift = 0  # default shift
        if training_window_type == TrainingWindowType.EXPANDING:
            output_chunk_length = max_horizon
            # output_chunk_shift = 5
            input_chunk_length = (
                sum(series_target.time_index < time) - output_chunk_length
            )
            model.reform(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                output_chunk_shift=output_chunk_shift,
                likelihood=QuantileRegression(quantiles=quantiles),
            )
            horizon = min(max_horizon, sum(times >= time))
            adjust_time = output_chunk_shift
        elif training_window_type == TrainingWindowType.ROLLING:
            # output_chunk_length = max_horizon
            # output_chunk_shift = 5
            # model.reform(
            #     input_chunk_length=24,
            #     output_chunk_length=1,
            #     output_chunk_shift=output_chunk_shift,
            # )
            # adjust_time = 0  # output_chunk_shift
            pass
        else:
            raise ValueError("Unknown TrainingWindowType.")

        model.fit(
            ts_target,
            past_covariates=ts_covars if covar_list else None,
        )
        forecast = model.predict(
            horizon,
            series=ts_target[target_cols],
            past_covariates=ts_covars if covar_list else None,
            num_samples=num_samples,
        )

        # # Reframe forecast as pandas DataFrame
        # forecast_df = forecast.quantile(quantiles).to_dataframe()
        # forecast_df["horizon"] = list(range(1, len(forecast_df) + 1))
        # forecast_df = forecast_df.reset_index().melt(
        #     id_vars=["time", "horizon"],
        #     var_name="quantile",
        #     value_name="Cases",
        # )

        forecast_df = forecast.quantile(quantiles).to_dataframe().reset_index()
        forecast_df["horizon"] = list(range(1, len(forecast_df) + 1))
        df_long = forecast_df.melt(
            id_vars=["time", "horizon"],
            var_name="col",
            value_name="Cases",
        )
        df_long["region"] = df_long["col"].str.extract(r"Cases_([A-Za-z0-9]+)_")
        df_long["quantile"] = df_long["col"].str.extract(r"q([0-9.]+)").astype(float)
        forecast_df = df_long[["time", "region", "quantile", "horizon", "Cases"]]

        # Append predictions at timepoint
        print(f"Adjust time by {adjust_time} months")
        forecast_dfs.append(forecast_df)
        # forecast_dfs.append(
        #     pd.DataFrame(
        #         {
        #             "model": label,
        #             "time": forecast_df["time"],  # + pd.DateOffset(months=adjust_time),
        #             "horizon": forecast_df["horizon"],
        #             "quantile": forecast_df["quantile"],
        #             "Cases": forecast_df["Cases"],
        #         }
        #     )
        # )

    # Combine timepoints into a single DataFrame
    forecast_df = pd.concat(forecast_dfs)

    # Save forecast
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(
        output_path / f"forecast_{label}_{training_window_type.value}_{iso_name}.csv",
    )
