import argparse
import numpy as np
import pandas as pd
from forecasting.models import TrainingWindowType
from forecasting.run_model import run_model
from forecasting.plotting import plot_historical_forecast

# Set random seeds for reproducibility
np.random.seed(42)


def transform(x):
    # Transform used for data analysis
    return np.log1p(x)


def itransform(y):
    # (Inverse) transform (only used for plotting)
    return y  # np.expm1(y)


def construct_covariates(df):
    covar = pd.DataFrame(index=df.index)  # return df with matching indices
    covar["Lag_Cases_1"] = transform(df["Cases"]).shift(1)
    covar["spe03_2"] = df["spe03"].shift(2)
    covar["spa03_2"] = df["spa03"].shift(2)
    covar["ssta_2"] = df["ssta"].shift(2)
    return covar


def main():
    parser = argparse.ArgumentParser(description="Run forecasting model")

    # Functions
    parser.add_argument(
        "--run_model",
        action="store_true",
        help="Flag to run the forecasting model",
    )
    parser.add_argument(
        "--plot_forecast",
        action="store_true",
        help="Flag to plot historical forecasts",
    )

    # Arguments
    parser.add_argument(
        "--iso",
        type=str,
        default="DOM",
        help="ISO code for the country to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="n-hits",
        help="Model to use for forecasting",
    )
    parser.add_argument(
        "--training_window_type",
        type=str,
        default="expanding",
        help="Type of training window: expanding or rolling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    if args.run_model:
        run_model(
            iso=args.iso,
            model_name=args.model,
            training_window_type=TrainingWindowType(args.training_window_type),
            transform=transform,
            construct_covariates=construct_covariates,
            output_dir=args.output_dir,
        )
    if args.plot_forecast:
        plot_historical_forecast(
            label=args.model,
            training_window_type=TrainingWindowType(args.training_window_type),
            iso=args.iso,
            transform=transform,
            itransform=itransform,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    raise SystemExit(main())
