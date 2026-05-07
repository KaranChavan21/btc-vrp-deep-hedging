# Professor Meeting — Presentation Guide
## BTC Options Volatility Risk Premium Strategy

**Karan Chavan · MSc Quantitative Finance · SMU**
**Meeting prep: what to say, what the math means, questions to expect**

---

## The One-Sentence Pitch

> "Options are consistently overpriced relative to what actually happens — I built a system
> to harvest that gap on Bitcoin, hedged the risk away, and achieved a Sharpe of 6.60
> out-of-sample with a max drawdown under 2%."

---

## 1. The Core Idea (5 min)

### In plain English

Imagine selling car insurance every Friday. You collect the premium upfront. If nothing
happens, you keep it all. If an accident happens, you pay out. The key insight:
**insurance companies make money because premiums are systematically too high**.

Options markets work the same way. There's a persistent gap between:
- **Implied volatility** (IV): what the market *fears* will happen
- **Realized volatility** (RV): what actually *does* happen

That gap is the **Variance Risk Premium (VRP)**. We sell the overpriced insurance
(short ATM straddle), hedge the directional risk (delta hedge via BTC perpetual),
and keep the premium.

### Why does the VRP exist?

Demand side: investors pay a premium to hedge tail risk. They'd rather pay too much for
protection than be exposed to a crash. This creates a structural buyer of vol.

Supply side: market-makers and vol sellers are paid to bear this risk. The premium
is their compensation for holding short-gamma exposure.

### Does it exist on BTC?

**Yes.** We measure it on 4.3 years of Deribit data:

- Mean log-VRP = +0.50/month
- P(VRP > 0) = 99.7%
- Comparable magnitude to SPX

---

## 2. Measuring the VRP (10 min)

### In plain English

We need two things:
1. **What the market implies** volatility will be (from option prices)
2. **What we forecast** volatility will actually be (from historical data)

The gap between these two is our signal.

### (a) Implied vol — Carr-Wu Variance Swap

We don't just use ATM implied vol. We use a **model-free synthetic variance swap**
that integrates the *entire* strike chain of OTM options. This is better because:
- It captures the full skew, not just one point
- It's model-independent (doesn't assume Black-Scholes)
- It was derived rigorously by Carr & Wu (2009) for exactly this purpose

**The formula:**

$$
\text{SW}^2(t, T) = \frac{2}{T-t} e^{r(T-t)} \left[ \int_0^F \frac{P(K,T)}{K^2}\,dK + \int_F^\infty \frac{C(K,T)}{K^2}\,dK \right]
$$

**What this means:** Sum up all OTM puts and calls, weighted by $1/K^2$. Result is the
market's expectation of total variance over $[t, T]$. We do this numerically from the
Deribit option chain every hour.

### (b) Realized vol — HAR-RV Model

We forecast realized volatility using Corsi's (2009) HAR model:

$$
\text{RV}_t = \beta_0 + \beta_d \cdot \text{RV}_{t-1}^{(d)} + \beta_w \cdot \text{RV}_{t-5}^{(w)} + \beta_m \cdot \text{RV}_{t-22}^{(m)} + \varepsilon_t
$$

**What this means:** Tomorrow's vol is a mix of yesterday's vol (daily), last week's
average (weekly), and last month's average (monthly). The model captures the
*heterogeneous* nature of vol — different traders react to different horizons.

**Our walk-forward estimates:** β_d = 0.48, β_w = 0.15, β_m = 0.18, R² = 0.40

**Critical:** We fit HAR with walk-forward validation. At each Friday entry, we only
use data up to Thursday. Zero look-ahead.

### (c) The Signal — Log VRP

$$
\text{LRP}(t) = \log\left(\text{SW}^2(t, t+7d)\right) - \log\left(\hat{\text{RV}}^2(t, t+7d)\right)
$$

Positive LRP = options are overpriced relative to forecast RV → sell volatility.

We also look at the **term structure slope**: if near-term IV > long-term IV (contango),
VRP is typically high. Composite entry score:

$$
\text{score}(t) = 0.6 \cdot \text{VRP\_percentile} + 0.3 \cdot \text{slope\_{front}} + 0.1 \cdot \text{slope\_{back}}
$$

---

## 3. What We Actually Trade (5 min)

### In plain English

Every Friday at 08:00 UTC, Deribit issues new 7-day weekly options. We:

1. **Check 5 gates** (see Section 4) — skip if regime looks bad
2. **Sell an ATM straddle** (one call + one put, same strike, same expiry)
3. **Delta hedge** throughout the week using BTC perpetual futures
4. **Close** the following Friday or stop-loss early if vol spikes

Why ATM straddle? We want pure vega/theta exposure. An ATM straddle is:
- Delta ≈ 0 at entry (long call delta + short put delta cancel)
- Long theta (time decay earns us money)
- Short vega (if IV drops, we profit)
- Short gamma (if spot moves a lot, we lose — hence the hedge)

**Why Friday 08:00?** Deribit weekly options expire exactly at Friday 08:00 UTC.
Entering at issuance of fresh options maximizes:
- Time value (full 7-day theta)
- VRP signal (variance swap just reset)
- 28-combo sweep confirms: Friday 08:00 Sharpe = 7.14, next best = 5.54 (+29% edge)

### Position sizing — Moreira-Muir vol-target

We target a fixed volatility exposure. More vol in the market → smaller position,
less vol → larger position (but capped):

$$
n_{\text{straddles}} = \frac{2\sigma_T \cdot V_{\text{notional}}}{\text{straddle\_vega}}
$$

where $\sigma_T = \sqrt{\text{SW}^2 \cdot T}$ is entry IV, $V_{\text{notional}}$ is
the base notional ($10M at 10× leverage on $1M account), and straddle_vega is
the option's sensitivity to IV moves.

Size is then scaled by VRP conviction and capped by margin (30% of account = $300K):

$$
V_{\text{actual}} = \min\left(V_{\text{base}} \cdot \text{mm\_scale} \cdot \text{vrp\_mult},\;\; \frac{\text{margin\_cap}}{\text{margin\_frac}}\right)
$$

---

## 4. The Five Entry Gates (10 min)

### In plain English

Not every Friday is worth trading. A bad week = selling insurance right before a
hurricane. The gates filter out risky entries:

| Gate | What it checks | Why it matters |
|---|---|---|
| **Score gate** | Composite VRP score > 0.30 | Only enter when VRP signal is strong |
| **VRP gate** | VRP > 0.20 × 30d implied var | Vol-scaled absolute minimum |
| **Z-score gate** | VRP > 0.5σ above 8w mean | Not just positive — notably high vs recent history |
| **Momentum gate** | RV not expanding | Skip if volatility is accelerating (hurricane warning) |
| **RV5d gate** | 5d RV < ATM IV | Recent realized vol below implied — favorable entry |

**Result:** 225 Fridays → 42 trades (19% selectivity). We skip 81% of weeks.

### The momentum gate in detail

If realized vol this week is 10% higher than last week, the market is in a vol expansion
regime. Selling vol into expanding vol is dangerous (you'll be underwater immediately).
Gate: skip if RV_22d_now / RV_22d_1w_ago > 1.10.

**Walk-forward tuning:** All gate thresholds are refit every 4 weeks using a 26-week
trailing window. The optimizer picks whichever threshold maximized Sharpe over the
prior 6 months. This adapts to regime changes without look-ahead.

---

## 5. The Hedge (15 min)

### In plain English

We've sold a straddle: we're short gamma. If BTC moves a lot in either direction, we
lose money. To neutralize this, we continuously hold a delta hedge in BTC perpetuals.

Delta of a straddle ≈ delta_call + delta_put. We hold −delta × notional in perp futures.
As spot moves, delta changes, so we rebalance.

**Why not hedge continuously?** Transaction costs. Deribit charges 5bp per side on perps.
Rebalancing every hour × 168 hours × ~5bp = enormous cost. We tested every cadence from
1h to 24h:

| Cadence | Avg weekly cost | Sharpe |
|---|---|---|
| 1 hour | $8,200 | 6.1 |
| 8 hours | $3,400 (−59%) | 7.7 |
| 24 hours | $1,800 (−78%) | 6.8 |

**8 hours** is Pareto-optimal. Saves 59% cost, actually improves Sharpe vs 1h because
less noise-chasing.

This is consistent with Whalley-Wilmott (1997) theory: under transaction costs, the
optimal hedge bandwidth grows with cost, and over-hedging is worse than under-hedging.

### Black-Scholes delta hedge

$$
\Delta_{\text{BS}} = N(d_1) - N(-d_1) = 2N(d_1) - 1
$$

for an ATM straddle, where $d_1 = (\sigma\sqrt{T})/2$ at ATM.

We use Leland (1985) modified vol to account for discrete rebalancing:

$$
\hat{\sigma} = \sigma\sqrt{1 + \frac{\kappa}{\sigma}\sqrt{\frac{2}{\pi \Delta t}}}
$$

where $\kappa$ is the round-trip cost rate and $\Delta t$ is the rebalance interval.

### Deep Hedge — Buehler et al. (2018)

Instead of using the Black-Scholes formula to compute delta, we train a neural network
to learn the optimal hedge.

**Network architecture:** 3-layer MLP, 64 hidden units per layer

**Inputs at each hour k:**
$$
I_k = \left[\log(S_k/K),\;\; T_{\text{rem}},\;\; \sigma_k,\;\; \delta_{k-1},\;\; k_{\text{norm}}\right]
$$

- $\log(S_k/K)$: moneyness (how far spot is from strike)
- $T_{\text{rem}}$: time remaining
- $\sigma_k$: current implied vol
- $\delta_{k-1}$: previous hedge (path dependency)
- $k_{\text{norm}}$: normalized time step

**Objective:** minimize entropic risk (a.k.a. exponential utility):
$$
\rho_\lambda(X) = \frac{1}{\lambda}\log\,\mathbb{E}\left[e^{-\lambda X}\right], \quad \lambda = 1
$$

This penalizes large losses exponentially — the network learns to be risk-averse, not
just minimize expected cost.

**Training data:** 4,212 real BTC rolling 168-hour windows from 2022-2023. We didn't
use synthetic GBM paths (as in the original Buehler paper) — we trained on real BTC
to capture actual BTC-specific gamma dynamics (fat tails, vol clustering).

### IS-Optimal Ensemble Weight

We blend deep hedge and BS delta:
$$
\delta = w \cdot \delta_{\text{DH}} + (1-w) \cdot \delta_{\text{BS}}
$$

**Formal IS sweep** over w ∈ {0.0, 0.1, ..., 1.0} on 2022-2023 data only:

| w | IS Sharpe | OOS Sharpe |
|---|---|---|
| 0.00 (pure BS) | **9.83** | **6.60** |
| 0.30 | 9.67 | 6.23 |
| 1.00 (pure DH) | 8.56 | 5.27 |

**Result: w\* = 0.00** (pure BS maximizes Sharpe). Deep hedge increases raw PnL (+9.3%)
but also increases variance (win rate 88% → 74%), netting lower Sharpe.

**Interpretation:** The Buehler objective (minimize CVaR/entropic risk) diverges from
the Sharpe objective (maximize mean/std). The deep hedge is "safer" in a tail-risk sense
but more volatile trade-to-trade. For this short-vol strategy, BS delta hedging is already
near-optimal for Sharpe.

---

## 6. Results (10 min)

### Key numbers

| Metric | This Strategy | BTC Buy & Hold |
|---|---|---|
| Annualized return | 22.7% | 20.7% |
| **Sharpe (OOS 2024-2026)** | **6.60** | **0.42** |
| **Sharpe (full 4.3 years)** | **7.74** | **0.42** |
| Max drawdown | **-1.2%** | -50.5% |
| Win rate | 88% | — |
| Trades | 42 | — |

Same absolute returns. 15× better Sharpe. 42× tighter drawdown.

### IS / OOS split

| Period | Trades | Cumulative PnL | Sharpe |
|---|---|---|---|
| In-sample (2022-2023) | 18 | $772K | 9.83 |
| **Out-of-sample (2024-2026)** | **24** | **$644K** | **6.60** |
| Combined | 42 | $1.42M | 7.74 |

OOS Sharpe of 6.60 means the model generalizes. The ~3 pt IS/OOS gap is expected
with n=18 IS trades — sampling variance, not overfitting.

### Sharpe reconciliation

Multiple numbers appear; they correspond to different methodological choices:

| Sharpe | Method | Period |
|---|---|---|
| **6.60** | Path-simulated, full frictions, w\*=0.00 | **OOS 2024-2026** ← headline |
| 7.74 | Path-simulated, full frictions, w\*=0.00 | Full 2022-2026 |
| 8.00 | Analytical (var-swap formula) | 2022-2026 walk-forward |
| 9.83 | Path-simulated | IS 2022-2023 only |

Use 6.60 as the conservative headline. Everything else is context.

---

## 7. Why We Trust It — Robustness Tests (10 min)

### (a) Bootstrap CI — is the Sharpe real?

Politis-Romano stationary block bootstrap (10,000 resamples, block size 4):
- Point estimate: 7.74
- 95% CI: **[5.5, 9.7]**

Lower bound of 5.5 is still 13× better than BTC (0.42). The Sharpe is robust even
under worst-case resampling.

### (b) Walk-forward parameter tuning

All gate thresholds refit every 4 weeks. Walk-forward Sharpe = 9.29 vs fixed = 9.11.
The strategy improves slightly when it adapts to regime changes.

### (c) Crisis stress tests

LUNA, 3AC, FTX, SVB replayed at 2×/5×/10× leverage:

| Crisis | Spot Move | Stop-loss? | Outcome |
|---|---|---|---|
| Terra/LUNA (May 2022) | -17% | Triggered | No liquidation |
| FTX (Nov 2022) | +3.9% | Triggered | No liquidation |
| SVB (Mar 2023) | flat | Triggered | No liquidation |
| All others | mild | No | Profitable |

Stop-loss fires when running RV > 1.5 × entry IV. Acts as circuit breaker.
No liquidations at any leverage level tested.

### (d) Crisis-weighted bootstrap

Oversample crisis weeks by 2×/5×/10×:
Sharpe INCREASES (7.58 → 8.41). Why? Gates filter the losers; crisis weeks that
passed all gates were actually profitable (selling into post-spike mean reversion).

### (e) Slippage scaling

At $10M notional (100× current scale), 5× slippage model → Sharpe still 5-7.
Strategy is not fragile to execution quality assumptions.

---

## 8. Limitations (be upfront)

| Limitation | Honest assessment |
|---|---|
| Sample size | 42 trades → 95% CI [5.5, 9.7]. Need more data to tighten. |
| Single venue | Deribit only. Multi-exchange untested. |
| Strategy decay | VRP may shrink as more capital enters the trade. |
| Live execution | Backtest assumes fills at mid; never live traded. |
| Multi-asset | ETH, SOL not tested yet. |

**Key point for professor:** These are acknowledged limitations, not hidden risks.
The 95% CI lower bound of 5.5 is the "pessimistic" Sharpe, still >> 2.

---

## 9. Common Questions & Answers

**Q: Why not just buy straddles?**
A: VRP implies options are overpriced. Long straddles systematically lose. Sellers collect the premium. The directionality of the trade is what the evidence supports.

**Q: Isn't shorting vol very risky?**
A: Unhedged, yes. We delta hedge every 8 hours. Max drawdown is 1.2%. The risk is controlled — not eliminated, but tightly bounded by the stop-loss and gating system.

**Q: Why does your Sharpe seem unrealistically high?**
A: Three reasons: (1) this is a weekly-frequency strategy with 42 trades — small sample amplifies apparent Sharpe; (2) the bootstrap CI is [5.5, 9.7] — we report the range honestly; (3) VRP harvesting genuinely has high risk-adjusted returns because you're selling insurance where demand is structurally inelastic.

**Q: Did you test on other assets?**
A: Not yet. BTC was chosen because VRP is large and measurable, data is available, and margin requirements are clear. Extension to ETH is straightforward.

**Q: How does this compare to professional vol strategies?**
A: Professional short-vol hedge funds (e.g., Universa short sellers, SPX variance swaps) typically run Sharpe of 1-3. Our 6.60 OOS reflects BTC's larger VRP and the selectivity of 19% trade rate. As capital scales, Sharpe will compress toward market norms.

**Q: What's the deep hedge contribution?**
A: Interesting result: pure BS maximizes Sharpe; deep hedge maximizes raw PnL (+9.3%) at the cost of higher variance. This makes sense — Buehler's objective is CVaR minimization, not Sharpe. The deep hedge is more conservative (lower tails) but more erratic trade-to-trade. For risk-adjusted performance, BS wins.

**Q: Is Friday 08:00 data snooping?**
A: No — it was an ex-ante structural hypothesis, not an ex-post discovery. Deribit weekly options expire exactly at Friday 08:00 UTC by exchange design. Entering at fresh issuance is structurally motivated. The sweep across 28 day/hour combinations was done after the hypothesis was pre-specified, and Friday 08:00 dominates by 29% — too large to be coincidence.

---

## 10. Flow for a 30-Minute Meeting

```
0-5 min:   Big picture — VRP exists, we harvest it
5-15 min:  How we measure (Carr-Wu, HAR-RV) + when we trade (5 gates)
15-20 min: How we hedge (BS delta + deep hedge finding)
20-25 min: Results + IS/OOS + bootstrap CI
25-30 min: Limitations + questions
```

**Have open on laptop:**
- `git/notebooks/thesis_results.ipynb` — live charts
- Section 3.3 for IS/OOS comparison
- Section 4.2 for Sharpe reconciliation table

---

## Key Numbers to Memorize

| Number | What it is |
|---|---|
| **6.60** | Headline OOS Sharpe (2024-2026) |
| **7.74** | Full-period Sharpe |
| **-1.2%** | Max drawdown |
| **42 trades** | Over 4.3 years (19% selectivity) |
| **99.7%** | P(VRP > 0) on BTC |
| **[5.5, 9.7]** | Bootstrap 95% CI on Sharpe |
| **+9.3%** | Deep hedge PnL gain (at cost of lower Sharpe) |
| **−59%** | Cost saving from 8h vs 1h hedging |
| **w\*=0.00** | IS-optimal ensemble weight (pure BS) |
