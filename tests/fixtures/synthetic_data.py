import numpy as np
import pandas as pd


def make_residual_signal_data(
    n: int = 1000,
    random_state: int = 42,
    include_interaction: bool = True,
    heterogeneity: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x_noise = rng.normal(size=n)

    sector_values = np.array(["healthcare", "industrial", "software", "consumer"])
    vintage_values = np.arange(1995, 2024)
    sector = pd.Series(rng.choice(sector_values, size=n), name="sector")
    vintage_year = pd.Series(rng.choice(vintage_values, size=n), name="vintage_year")

    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x_noise": x_noise,
        }
    )

    base_signal = (0.55 * x2) - (0.35 * x3) + (0.15 * x_noise)
    y_pred = pd.Series(base_signal + rng.normal(scale=0.25, size=n), name="y_pred")

    nonlinear_x1 = 0.70 * np.sin(1.4 * x1) + 0.22 * (np.square(x1) - 1.0)
    interaction = 0.40 * x2 * x3 if include_interaction else 0.0
    heterogeneity_effect = np.zeros(n)
    if heterogeneity:
        sector_effect = sector.map(
            {
                "healthcare": 0.18,
                "industrial": -0.12,
                "software": 0.14,
                "consumer": -0.08,
            }
        ).to_numpy()
        vintage_effect = 0.025 * (vintage_year.to_numpy() - float(vintage_year.mean()))
        heterogeneity_effect = sector_effect + vintage_effect

    residual = nonlinear_x1 + interaction + heterogeneity_effect + rng.normal(scale=0.45, size=n)
    y_true = pd.Series(y_pred.to_numpy() + residual, name="y_true")

    sample_weight = pd.Series(
        0.75 + rng.gamma(shape=2.0, scale=0.25, size=n),
        name="sample_weight",
    )
    split_col = pd.Series(
        np.select(
            [np.arange(n) < int(0.70 * n), np.arange(n) < int(0.85 * n)],
            ["train", "validation"],
            default="holdout",
        ),
        name="split",
    )

    return X, y_true, y_pred, sample_weight, sector, vintage_year, split_col
