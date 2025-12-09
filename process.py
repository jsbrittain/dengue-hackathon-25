import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    AutoARIMA,
    TiDEModel,
    NBEATSModel,
    NHiTSModel,
)
from darts.utils.likelihood_models import QuantileRegression

quantiles = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]

iso = 'DOM'

dirstem = Path(__file__).parent
max_horizon = 6

# Load cases
df = pd.read_csv(dirstem / f"Covariates_data_{iso}.csv")
df["time"] = pd.to_datetime(df["time"]).dt.to_period("M")
df['time'] = df['time'].dt.to_timestamp()

models = {
    # "Drift": NaiveDrift(),
    # "Seasonal": NaiveSeasonal(K=12),
    # "AutoARIMA": AutoARIMA(),
    # "TiDE": TiDEModel(
    #     input_chunk_length=24,
    #     output_chunk_length=max_horizon,
    #     likelihood=QuantileRegression(quantiles=quantiles),
    # ),
    "N-HiTS": NHiTSModel(
        input_chunk_length=24,
        output_chunk_length=max_horizon,
        likelihood=QuantileRegression(quantiles=quantiles),
    )
    # "N-BEATS": NBEATSModel(
    #     input_chunk_length=24,
    #     output_chunk_length=max_horizon,
    #     likelihood=QuantileRegression(quantiles=quantiles),
    # ),
}
label = 'N-HiTS'

# Construct covariates
df['Lag_Cases_1'] = np.log1p(df['Cases']).shift(1)
df['spe03_2'] = df['spe03'].shift(2)
df['spa03_2'] = df['spa03'].shift(2)
df['ssta_2'] = df['ssta'].shift(2)

covar_list = ['Lag_Cases_1', 'spe03_2', 'spa03_2', 'ssta_2']

df = df[2:]


def transform(x):
    return np.log1p(x)


def itransform(y):
    return np.expm1(y)


# Specify target and covariates
target_col = 'Cases'

# Transform
df[target_col] = transform(df[target_col])

# Build DARTS TimeSeries
series = TimeSeries.from_dataframe(df, time_col='time', value_cols=['Cases', *covar_list])
series = series.astype(np.float32)

forecast_models = {}
for label, model in models.items():
    times = sorted(df['time'][(df['time'] >= '2018-01')])  #  & (df['time'] < '2021-01')])
    forecast_model = []
    for time in times:
        print(f"Fitting model {model} for time {time}")

        target = series.drop_after(time)
        cov = series.drop_after(time)

        # target=target[-24:]
        # cov=cov[-24:]

        # if label == 'TiDE':
        adjust_time = 0
        if True:
            output_chunk_length = max_horizon
            input_chunk_length = sum(series.time_index < time) - output_chunk_length
            model = TiDEModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                likelihood=QuantileRegression(quantiles=quantiles),
            )
            adjust_time = output_chunk_length

        model.fit(target[target_col], past_covariates=cov[covar_list])
        forecast = model.predict(6, num_samples=1000)

        # bt = model.historical_forecasts(
        #     series=target[target_col],
        #     past_covariates=cov[covar_list],
        #     forecast_horizon=max_horizon,
        #     # start=start_date
        #     stride=1,
        #     retrain=True,
        #     last_points_only=False,  # this changes the output format
        #     verbose=False,
        # )

        forecast_df = forecast.quantile(quantiles).to_dataframe()
        forecast_df['horizon'] = list(range(1, len(forecast_df) + 1))
        ts = forecast_df.index
        forecast_df = forecast_df.reset_index().melt(
            id_vars=['time', 'horizon'],
            var_name='quantile',
            value_name='Cases',
        )

        forecast_model.append(pd.DataFrame({
            'model': label,
            'time': forecast_df['time'] + pd.DateOffset(months=-adjust_time),
            'horizon': forecast_df['horizon'],
            'quantile': forecast_df['quantile'],
            'Cases': forecast_df['Cases'],
        }))
    forecast_models[label] = pd.concat(forecast_model)

horizons = [1, 3, 6]
plt.figure()
for ix, horizon in enumerate(horizons):
    plt.subplot(len(horizons), 1, ix + 1)
    plt.plot(itransform(series.to_dataframe()['Cases']), label='Original Series')
    df = forecast_models[label][forecast_models[label]['horizon'] == horizon]
    plt.fill_between(
        df[df["quantile"] == "Cases_q0.500"]['time'],
        itransform(df[df["quantile"] == "Cases_q0.050"]['Cases']),
        itransform(df[df["quantile"] == "Cases_q0.950"]['Cases']),
        color="steelblue",
        alpha=0.3,
    )
    plt.plot(
        df[df['quantile'] == 'Cases_q0.500']['time'],
        itransform(df[df['quantile'] == 'Cases_q0.500']['Cases']),
        label=f'Forecast Horizon {horizon} - {label}',
    )
plt.show()
