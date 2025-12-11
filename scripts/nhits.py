import numpy as np
import pandas as pd
from forecasting.models import TrainingWindowType
from forecasting.run_model import run_model
from forecasting.plotting import plot_historical_forecasts


# ======================================================================================
# Parameters
# ======================================================================================


isos = ["BLM", "DOM", "MAF", "BRB", "GRD", "PRI"]
# model = "N-HiTS"
model = "Chronos2"
training_window_type = TrainingWindowType.ROLLING
output_dir = "outputs"

steps = {}
steps["run_model"] = True
steps["plot_historical_forecasts"] = True


# ======================================================================================
# Model-specific functions
# ======================================================================================


# Transform used for data analysis
def transform(x):
    return np.log1p(x)


# (Inverse) transform (only used for plotting)
def itransform(y):
    return y  # np.expm1(y)


# Distance between islands (in km)
dist = {
    ("DOM", "PRI"): 390,
    ("PRI", "DOM"): 390,
    ("DOM", "BRB"): 1600,
    ("BRB", "DOM"): 1600,
    ("DOM", "GRD"): 1800,
    ("GRD", "DOM"): 1800,
    ("DOM", "MAF"): 650,
    ("MAF", "DOM"): 650,
    ("DOM", "BLM"): 670,
    ("BLM", "DOM"): 670,
    ("PRI", "BRB"): 1300,
    ("BRB", "PRI"): 1300,
    ("PRI", "GRD"): 1400,
    ("GRD", "PRI"): 1400,
    ("PRI", "MAF"): 270,
    ("MAF", "PRI"): 270,
    ("PRI", "BLM"): 290,
    ("BLM", "PRI"): 290,
    ("BRB", "GRD"): 260,
    ("GRD", "BRB"): 260,
    ("BRB", "MAF"): 650,
    ("MAF", "BRB"): 650,
    ("BRB", "BLM"): 670,
    ("BLM", "BRB"): 670,
    ("GRD", "MAF"): 780,
    ("MAF", "GRD"): 780,
    ("GRD", "BLM"): 800,
    ("BLM", "GRD"): 800,
    ("MAF", "BLM"): 35,
    ("BLM", "MAF"): 35,
}


def construct_covariates(df: pd.DataFrame) -> pd.DataFrame:
    covar = pd.DataFrame(index=df.index)  # return df with matching indices
    match 3:
        case 0:  # No covariates
            return covar
        case 1:  # Specific lagged covariates
            covar["Lag_Cases_1"] = transform(df["Cases"]).shift(1)
            covar["spe03_2"] = df["spe03"].shift(2)
            covar["spa03_2"] = df["spa03"].shift(2)
            covar["ssta_2"] = df["ssta"].shift(2)
        case 2:  # Top 5 principal components of lagged covariates
            from sklearn.decomposition import PCA

            covar_cols = df.columns.difference(["time", "Cases"])
            lagged_covar = df[covar_cols].shift(1)
            pca = PCA(n_components=5)
            pcs = pca.fit_transform(lagged_covar.dropna())
            for i in range(pcs.shape[1]):
                covar[f"PC_{i + 1}"] = np.nan
                covar.loc[lagged_covar.dropna().index, f"PC_{i + 1}"] = pcs[:, i]
        case 3:  # Directional human mobility (flow) proxy
            for iso_from in isos:
                for iso_to in isos:
                    if iso_from != iso_to:
                        col_name = f"flow_{iso_from}_to_{iso_to}"
                        dst = dist[(iso_from, iso_to)]
                        covar[col_name] = np.log(df[f"pop_{iso_from}"]) - np.log(dst)
        case _:
            raise ValueError("Invalid covariate construction case")
    return covar


# ======================================================================================
# Pipeline
# ======================================================================================


# Set random seeds for reproducibility
np.random.seed(42)


# Run the model, save results to the specified output folder
if steps["run_model"]:
    run_model(
        isos=isos,
        model_name=model,
        training_window_type=training_window_type,
        transform=transform,
        construct_covariates=construct_covariates,
        output_dir=output_dir,
    )

# Plot historical forecast (loads results from output folder)
if steps["plot_historical_forecasts"]:
    plot_historical_forecasts(
        label=model,
        training_window_type=training_window_type,
        isos=isos,
        transform=transform,
        itransform=itransform,
        output_dir=output_dir,
    )
