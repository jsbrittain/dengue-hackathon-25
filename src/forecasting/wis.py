import numpy as np
import pandas as pd


def compute_wis(df_pred, df_true, itransform=lambda x: x, quantiles=None):
    """
    df_pred: long format forecast dataframe with columns:
        time, region, quantile, horizon, Cases
    df_true: dataframe with columns: time, region, Cases  (true values)

    quantiles: sorted list of quantiles present in df_pred.
               If None, they will be inferred.

    Returns:
        DataFrame indexed by ["time", "region", "horizon"]
        with a column "WIS"
    """

    # Ensure quantiles are known
    if quantiles is None:
        quantiles = sorted(df_pred["quantile"].unique())

    # Central interval alphas:
    # Example: quantiles = [0.01, 0.05, 0.10, ..., 0.95, 0.99]
    # valid interval pairs are (q_low, q_high)
    qs = quantiles
    interval_pairs = [(qs[i], qs[-i - 1]) for i in range(len(qs) // 2)]
    alphas = [hi - lo for lo, hi in interval_pairs]  # widths

    # Merge prediction + truth
    merged = df_pred.merge(
        df_true, on=["time", "region"], how="inner", suffixes=("_pred", "_true")
    )
    merged["Cases_pred"] = itransform(merged["Cases_pred"])
    merged["Cases_true"] = itransform(merged["Cases_true"])

    # pivot quantiles wide for each (time, region, horizon)
    df_wide = merged.pivot_table(
        index=["time", "region", "horizon"], columns="quantile", values="Cases_pred"
    )

    # true values grouped the same way
    y_true = merged.groupby(["time", "region", "horizon"])["Cases_true"].first()

    # Calculate WIS entry by entry
    wis_list = []

    for idx, row in df_wide.iterrows():
        t, r, h = idx
        truth = y_true.loc[(t, r, h)]

        # median absolute error
        median = row[0.5]
        score_components = [abs(truth - median)]

        # interval scores
        for (q_lo, q_hi), alpha in zip(interval_pairs, alphas):
            lower = row[q_lo]
            upper = row[q_hi]

            interval_score = alpha * (max(lower - truth, 0) + max(truth - upper, 0))
            score_components.append(interval_score)

        wis = np.sum(score_components) / (len(alphas) + 1)

        wis_list.append({"time": t, "region": r, "horizon": h, "WIS": wis})

    return pd.DataFrame(wis_list)
