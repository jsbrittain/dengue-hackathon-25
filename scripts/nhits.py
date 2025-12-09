import numpy as np
import pandas as pd
from forecasting.models import TrainingWindowType
from forecasting.run_model import run_model
from forecasting.plotting import plot_historical_forecast


iso = "DOM"
model = "n-hits"
training_window_type = TrainingWindowType.EXPANDING
output_dir = "outputs"


# Transform used for data analysis
def transform(x):
    return np.log1p(x)


# (Inverse) transform (only used for plotting)
def itransform(y):
    return y  # np.expm1(y)


def construct_covariates(df: pd.DataFrame) -> pd.DataFrame:
    covar = pd.DataFrame(index=df.index)  # return df with matching indices
    covar["Lag_Cases_1"] = transform(df["Cases"]).shift(1)
    covar["spe03_2"] = df["spe03"].shift(2)
    covar["spa03_2"] = df["spa03"].shift(2)
    covar["ssta_2"] = df["ssta"].shift(2)
    return covar


# Set random seeds for reproducibility
np.random.seed(42)


# Run the model, save results to the specified output folder
run_model(
    iso=iso,
    model_name=model,
    training_window_type=training_window_type,
    transform=transform,
    construct_covariates=construct_covariates,
    output_dir=output_dir,
)

# Plot historical forecast (loads results from output folder)
plot_historical_forecast(
    label=model,
    training_window_type=training_window_type,
    iso=iso,
    transform=transform,
    itransform=itransform,
    output_dir=output_dir,
)
