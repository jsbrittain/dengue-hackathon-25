import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path

from darts import TimeSeries
from darts.models import NaiveSeasonal

iso = 'BRB'

dirstem = Path(__file__).parent / 'data' / 'small_island' / iso

# Load cases
cases = pd.read_csv(dirstem / f"{iso}_epi_training.csv")
cases['Date'] = pd.to_datetime(cases['Month'])

print(cases.columns)

series = TimeSeries.from_dataframe(cases, time_col='Date', value_cols=['Cases'])

# Load covariates
ds = xr.open_dataset(str(dirstem / f"{iso}-reanalysis_monthly.zs.nc"))
ds

if False:
    # Naive Seasonal
    naive_model = NaiveSeasonal(K=52)
    naive_model.fit(series)
    naive_forecast = naive_model.predict(36)

    series.plot(label="actual")
    naive_forecast.plot(label="naive forecast (K=1)")
    plt.show()
