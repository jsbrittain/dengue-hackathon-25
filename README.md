# dengue-hackathon-25
Dengue forecasting hackathon 2025

Expects data to be located in a /data/ folder in the root directory with the following structure:

```
├── global
│   ├── elninossta.nc
│   └── pandemic_phase.nc
└── small_island
    ├── BRB
    │   ├── BRB-reanalysis_monthly.zs.nc
            ...
```

Data dictionaries: https://github.com/kraemer-lab/global_dengue_forecasting/blob/main/docs/README.md

Summary

| File | Description |
|------|-------------|
| (global) elninossta.nc | El Niño sea surface temperature anomalies |
| (global) pandemic_phase.nc | Global COVID-19 pandemic phases |
| epi_training.csv | Epidemiological training data |
| gdp_pc.zs.nc | Domestic product per capita |
| population.zs.nc | Population count |
| reanalysis_monthly.zs.nc | Monthly averages of ERA5 reanalysis |
| seasonal_forecast_monthly.zs.nc | Climatic forecasts |
| spa01.zs.nc | 10-day SPI (precipitation index) 1-month accumulation |
| spa03.zs.nc | 10-day SPI (precipitation index) 3-month accumulation |
| spa06.zs.nc | SPI (precipitation index) 6-month accumulation |
| spa12.zs.nc | SPI (precipitation index) 12-month accumulation |
| spe01.zs.nc | 10-day SPEI (evapotranspiration index) 1-month accumulation |
| spe03.zs.nc | 10-day SPEI (evapotranspiration index) 3-month accumulation |
| spe06.zs.nc | SPEI (evapotranspiration index) 6-month accumulation |
| spe12.zs.nc | SPEI (evapotranspiration index) 12-month accumulation |
| surv_wmean.zs.nc | Surveillance and reporting capacity |
| swvl1.zs.nc | Soil water volume in layer 1 |
