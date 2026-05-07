"""Block bootstrap confidence intervals for Sharpe ratio and other PnL metrics.

Politis-Romano (1994) stationary block bootstrap with random block sizes
(geometric distribution). Preserves serial dependence in trade PnL stream.

Usage:
    from src.layer6.bootstrap_ci import bootstrap_sharpe
    ci = bootstrap_sharpe(pnl_array, n_boot=10000, mean_block_size=4)
"""
from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path

LAYER6_OUT = Path("/Volumes/KC DRIVE/Paper Test/Deep Hedging/outputs/layer6")


def stationary_bootstrap_indices(n: int, mean_block_size: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: at each step, with prob 1/L start
    a new block; otherwise extend current block by 1."""
    p_new_block = 1.0 / mean_block_size
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p_new_block:
            indices[t] = rng.integers(0, n)
        else:
            indices[t] = (indices[t - 1] + 1) % n
    return indices


def bootstrap_sharpe(
    pnl: np.ndarray,
    n_boot: int = 10000,
    mean_block_size: float = 4.0,
    annualization: float = 52.0,
    seed: int = 42,
) -> dict:
    """Block-bootstrap Sharpe ratio + key stats. Returns mean, std, [2.5%, 97.5%] CI."""
    pnl = np.asarray(pnl, dtype=np.float64)
    n = len(pnl)
    if n < 4:
        return {"n": n, "error": "too few trades"}

    rng = np.random.default_rng(seed)

    sharpes = np.empty(n_boot)
    cums = np.empty(n_boot)
    win_rates = np.empty(n_boot)
    worsts = np.empty(n_boot)

    for i in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block_size, rng)
        sample = pnl[idx]
        s_std = sample.std()
        sharpes[i] = sample.mean() / s_std * np.sqrt(annualization) if s_std > 0 else np.nan
        cums[i] = sample.sum()
        win_rates[i] = (sample > 0).mean()
        worsts[i] = sample.min()

    sharpes = sharpes[~np.isnan(sharpes)]
    return {
        "n_trades": n,
        "n_boot": n_boot,
        "mean_block_size": mean_block_size,
        "sharpe": {
            "point": float(pnl.mean() / pnl.std() * np.sqrt(annualization)),
            "boot_mean": float(sharpes.mean()),
            "boot_std": float(sharpes.std()),
            "ci_2.5": float(np.quantile(sharpes, 0.025)),
            "ci_97.5": float(np.quantile(sharpes, 0.975)),
            "ci_5": float(np.quantile(sharpes, 0.05)),
            "ci_95": float(np.quantile(sharpes, 0.95)),
        },
        "cum_pnl": {
            "point": float(pnl.sum()),
            "boot_mean": float(cums.mean()),
            "ci_2.5": float(np.quantile(cums, 0.025)),
            "ci_97.5": float(np.quantile(cums, 0.975)),
        },
        "win_rate": {
            "point": float((pnl > 0).mean()),
            "ci_2.5": float(np.quantile(win_rates, 0.025)),
            "ci_97.5": float(np.quantile(win_rates, 0.975)),
        },
        "worst_trade": {
            "point": float(pnl.min()),
            "boot_mean": float(worsts.mean()),
            "ci_2.5": float(np.quantile(worsts, 0.025)),
            "ci_97.5": float(np.quantile(worsts, 0.975)),
        },
    }


def report_both_backtests() -> None:
    """Run bootstrap CI on analytical and real backtest PnL streams."""
    print("=" * 70)
    print("Block Bootstrap CI (Politis-Romano stationary, mean block=4 trades)")
    print("=" * 70)

    for label, path in [
        ("ANALYTICAL", LAYER6_OUT / "weekly_pnl.parquet"),
        ("REAL (path-sim, full frictions)", LAYER6_OUT / "weekly_pnl_real.parquet"),
    ]:
        if not path.exists():
            print(f"\n[skip] {path.name} missing")
            continue
        df = pl.read_parquet(path)
        if "enter" in df.columns:
            df = df.filter(pl.col("enter"))
        pnl = df["pnl"].to_numpy()
        ci = bootstrap_sharpe(pnl, n_boot=10000, mean_block_size=4.0)

        print(f"\n--- {label} (n={ci['n_trades']} trades) ---")
        s = ci["sharpe"]
        print(f"Sharpe (52w ann)")
        print(f"  point:     {s['point']:.2f}")
        print(f"  boot mean: {s['boot_mean']:.2f}  (std {s['boot_std']:.2f})")
        print(f"  95% CI:    [{s['ci_2.5']:.2f}, {s['ci_97.5']:.2f}]")
        print(f"  90% CI:    [{s['ci_5']:.2f}, {s['ci_95']:.2f}]")

        c = ci["cum_pnl"]
        print(f"Cum PnL")
        print(f"  point:     ${c['point']:>9,.0f}")
        print(f"  95% CI:    [${c['ci_2.5']:>8,.0f}, ${c['ci_97.5']:>9,.0f}]")

        w = ci["win_rate"]
        print(f"Win rate")
        print(f"  point:     {w['point']:.0%}")
        print(f"  95% CI:    [{w['ci_2.5']:.0%}, {w['ci_97.5']:.0%}]")

        wt = ci["worst_trade"]
        print(f"Worst trade")
        print(f"  point:     ${wt['point']:>9,.0f}")
        print(f"  95% CI:    [${wt['ci_2.5']:>9,.0f}, ${wt['ci_97.5']:>9,.0f}]")


if __name__ == "__main__":
    report_both_backtests()
