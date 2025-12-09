# Dengue Forecasting Hackathon 2025

**Small Island Nations, Machine Learning Team**

Data dictionaries: https://github.com/kraemer-lab/global_dengue_forecasting/blob/main/docs/README.md

# Pipeline

Target time-series and covariates are preprocessed using `notebooks/Preprocessing_climate_data_hackton_8dec.ipynb`. This produces `/data/Covariates_data_{iso}.csv` files for each country ISO code.

# Usage

Run in a Python virtual environment, we recommend [`uv`](https://docs.astral.sh/uv/getting-started/installation/). This code was developed with Python 3.13.2.

```
uv venv
uv pip install -e .
source .venv/bin/activate
```

Then, launch a forecast, e.g. for the example NHiTS model:
```
python scripts/nhits.py
```
