# Strategy Walkthrough — Plain English + Math
## BTC Options Variance Risk Premium Strategy

*Audience: someone with intermediate quant knowledge (~30%). I'll explain every formula
before showing it, name what each paper contributes, and stay practical throughout.*

---

## Part 1: The Big Picture

### What we're trying to do

Sell overpriced volatility on BTC options. Every Friday at 08:00 UTC, Deribit issues
fresh 7-day options. We sell an at-the-money straddle (one call + one put), hedge the
direction risk in BTC perpetuals, and pocket the premium decay.

### Why this works (the economic argument)

Option prices imply a future volatility level. Most of the time, this implied vol is
**higher** than what actually happens. The gap is called the **Variance Risk Premium
(VRP)**. It exists because:

- People want crash protection and pay too much for it (demand-side)
- Sellers need compensation for holding short-gamma exposure (supply-side)

Like an insurance company: collect premiums every week, pay out occasionally, profit on average.

### The three things we need to do well

1. **Measure VRP correctly** — quantify the edge
2. **Pick good entry weeks** — skip when vol is about to explode
3. **Hedge cheaply and effectively** — preserve the edge after costs

Each paper below contributes to one of these three problems.

---

## Part 2: Measuring VRP — The Two Sides of the Trade

To measure VRP, we need:
- **Implied variance** (what option prices say) — Carr-Wu
- **Forecast realized variance** (what we expect) — HAR-RV / Corsi

The difference is our edge.

---

### Paper 1 — Carr & Wu (2009): "Variance Risk Premiums"

**What we use:** model-free synthetic variance swap replication.

**The problem it solves:** ATM implied volatility is a single point. But options have a
"smile" — different strikes have different implied vols. Using only ATM IV throws away
information.

**The solution:** integrate the entire option chain to get the market's expectation of
total variance, with no model assumptions.

#### The formula

$$
\text{SW}^2(t, T) = \frac{2 e^{r(T-t)}}{T-t} \left[ \int_0^F \frac{P(K, T)}{K^2}\, dK + \int_F^\infty \frac{C(K, T)}{K^2}\, dK \right]
$$

#### Reading this formula in English

- $\text{SW}^2$ = the synthetic variance swap rate (annualized variance over $[t, T]$)
- $F$ = forward price of the underlying
- $P(K, T)$ = price of OTM put at strike $K$
- $C(K, T)$ = price of OTM call at strike $K$
- $1/K^2$ = a weight that gives more weight to OTM puts (which contain crash info)
- $e^{r(T-t)}$ = discount factor (since options pay at expiry)

**In one sentence:** Add up all OTM option prices, weighted by $1/\text{strike}^2$,
then scale. That's the market's variance expectation.

#### How we implement it

For each Friday at entry time, we read every traded OTM option from Deribit's chain,
fit an SVI surface (next paper) to smooth out noise, and numerically integrate.

**Output:** $\text{SW}^2_{7d}$ and $\text{SW}^2_{30d}$ — variance swap rates at 7-day
and 30-day horizons. We also derive **ATM IV** from these:

$$
\sigma_{\text{ATM}, 7d} = \sqrt{\text{SW}^2_{7d}}
$$

---

### Paper 2 — Gatheral & Jacquier (2014): "Arbitrage-Free SVI Volatility Surfaces"

**What we use:** SVI parameterization for smoothing the option chain.

**The problem it solves:** the raw option chain has gaps, illiquid points, and noise.
You need a smooth, arbitrage-free surface to integrate cleanly.

**The solution:** fit a 5-parameter functional form to the implied vol surface.

#### The formula (raw SVI)

$$
w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right]
$$

#### What the parameters mean (intuition)

- $a$ = overall level (vertical shift)
- $b$ = how fast wings grow (slope of smile)
- $\rho$ = skew (negative = more expensive puts, like equities)
- $m$ = where the smile bottom is (often ATM)
- $\sigma$ = curvature near ATM (smoothness)
- $k = \log(K/F)$ = log-moneyness
- $w$ = total implied variance ($\sigma^2 \cdot T$)

**In one sentence:** It's a hyperbola-like curve that fits implied variance vs strike
with five tunable knobs, with constraints to prevent arbitrage.

We solve $\arg\min$ of squared error vs market quotes for these five params at every
hour. Output: a clean $\sigma(K, T)$ surface to integrate.

---

### Paper 3 — Corsi (2009): "HAR Model of Realized Volatility"

**What we use:** the HAR-RV regression for forecasting realized variance.

**The problem it solves:** we need an estimate of *future* realized variance to compare
against implied. Naive ARMA models fail; vol exhibits long memory.

**The solution:** a simple OLS regression with three lagged variables — daily, weekly,
monthly RV. Captures the heterogeneous behavior of traders at different horizons.

#### The formula

$$
\text{RV}_{t+1} = \beta_0 + \beta_d \cdot \text{RV}_t^{(d)} + \beta_w \cdot \overline{\text{RV}}_{t}^{(w)} + \beta_m \cdot \overline{\text{RV}}_{t}^{(m)} + \varepsilon_{t+1}
$$

#### What this says

- $\text{RV}_t^{(d)}$ = yesterday's realized variance (1-day)
- $\overline{\text{RV}}_{t}^{(w)}$ = average RV over last 5 days
- $\overline{\text{RV}}_{t}^{(m)}$ = average RV over last 22 days
- $\varepsilon$ = noise term

**In one sentence:** Tomorrow's vol is a weighted blend of yesterday's, last week's, and
last month's vol.

#### Why three horizons?

Different traders react to different timescales. Day traders → daily RV matters.
Swing traders → weekly. Long-term hedgers → monthly. The HAR model captures all three.

#### Our walk-forward fit

We refit the betas every 4 weeks using a 26-week trailing window. **Critical**: at
each Friday entry, we only use data up to Thursday. Zero look-ahead.

**Coefficients (full sample):** $\beta_d = 0.48,\, \beta_w = 0.15,\, \beta_m = 0.18$,
$R^2 = 0.40$

#### The VRP signal

Once we have implied (Carr-Wu) and forecast (HAR), the **log VRP** is:

$$
\text{LRP}(t) = \log(\text{SW}^2_{7d}) - \log(\hat{\text{RV}}^2_{7d})
$$

We use log because it's symmetric and stable when variance is small. Positive LRP
= options overpriced → sell vol.

---

## Part 3: When to Trade — The Five Gates

Even with positive VRP, not every Friday is good. Some weeks, vol is about to explode
(LUNA crash, FTX collapse). The gates filter these out.

| Gate | Math | Why |
|---|---|---|
| Score gate | $0.6 \cdot \text{VRP}_{\text{pct}} + 0.3 \cdot \text{slope}_{\text{front}} + 0.1 \cdot \text{slope}_{\text{back}} > 0.30$ | Composite signal strength |
| VRP gate | $\text{vrp}_t > 0.20 \cdot \text{SW}^2_{30d}$ | Vol-scaled minimum |
| Z-score | $(\text{vrp}_t - \mu_{8w}) / \sigma_{8w} > 0.5$ | Above recent mean |
| Momentum | $\text{RV}_{22d}^{(t)} / \text{RV}_{22d}^{(t-1w)} < 1.10$ | No vol expansion |
| RV5d | $\text{RV}_{5d} < \sigma_{\text{ATM}, 30d}$ | Recent realized below implied |

**Result:** 225 Fridays → 42 trades (19% selectivity).

---

## Part 4: How Much to Trade — Position Sizing

### Paper 4 — Moreira & Muir (2017): "Volatility-Managed Portfolios"

**What we use:** scale position size **inversely** with volatility.

**The problem:** if you trade fixed notional, your risk varies. In high-vol regimes
your dollar P&L swings double. In low-vol regimes you barely move. Bad Sharpe.

**The solution:** target a constant *risk* exposure, not constant notional. Scale
position by $\text{TARGET\_VOL} / \text{realized\_vol}$.

#### Our sizing formula

We want target volatility $\sigma_T = 50\%$ annualized. Compute:

$$
\text{mm\_scale} = \min\left(\frac{\sigma_T}{\text{RV}_{22d}},\, 2.0\right)
$$

Then number of straddles:

$$
n_{\text{straddles}} = \frac{2 \sigma_T \sqrt{T} \cdot V_{\text{notional}}}{\text{vega}_{\text{straddle}}}
$$

with $V_{\text{notional}} = V_{\text{base}} \cdot \text{mm\_scale} \cdot \text{vrp\_mult}$,
capped by margin limit.

**In one sentence:** Trade more when vol is calm, less when vol is wild — keeps risk
constant and improves Sharpe.

---

## Part 5: How to Hedge — The Hard Part

We sold a straddle. We're now short gamma — if BTC moves a lot in either direction,
we lose. To neutralize the directional risk, we hold a delta hedge in BTC perpetuals.
Each hour, recompute delta, adjust hedge.

Three problems to solve:
- **What's the delta?** (Black-Scholes vs deep hedge)
- **How often to rebalance?** (transaction costs vs tracking error)
- **How to handle the cost?** (Leland modification, Whalley-Wilmott bandwidth)

---

### Paper 5 — Black & Scholes (1973): The classic

**What we use:** delta as the hedge ratio.

For an ATM straddle with time $T$ remaining and vol $\sigma$:

$$
\Delta_{\text{straddle}} = N(d_1) - N(-d_1) = 2 N(d_1) - 1
$$

where $d_1 = \frac{\log(S/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}$.

**At-the-money** with $S = K$, $r = 0$: $d_1 = \sigma\sqrt{T}/2$, so delta starts near zero
but grows as spot moves.

**In one sentence:** $\Delta$ tells you "how many BTC do I need to short to neutralize
the option's directional move."

---

### Paper 6 — Bakshi & Kapadia (2003): "Delta-Hedged Gains"

**What we use:** decomposition of the P&L of a delta-hedged short option position.

#### The decomposition

For a delta-hedged short straddle, P&L over $[t, t+dt]$ is approximately:

$$
\text{PnL} \approx \theta \cdot dt + \frac{1}{2} \Gamma \cdot (dS)^2 + \mathcal{V} \cdot d\sigma
$$

#### What each Greek contributes

- **Theta ($\theta$)**: time decay — earns us money ($\theta < 0$ for long, but we're short, so we collect $|\theta|$ per day)
- **Gamma ($\Gamma$)**: convexity — costs us when spot moves a lot
- **Vega ($\mathcal{V}$)**: vol sensitivity — earns us when IV drops, costs when IV rises

#### What this means for VRP harvesting

If implied vol > realized vol (positive VRP), then:
- Theta accrues faster than gamma drains → profit on average
- Vega earns when IV mean-reverts down

This is the math behind **why VRP harvesting works**. It's not magic — it's an
explicit Greek-based decomposition.

---

### Paper 7 — Leland (1985): "Option Pricing with Transaction Costs"

**What we use:** the modified volatility for hedging under costs.

**The problem:** classical BS assumes continuous hedging at zero cost. Real markets
charge per trade. Naive BS delta over-hedges → trading costs eat all profits.

**The solution:** inflate the volatility used in delta computation to under-hedge slightly.

#### The formula

$$
\hat{\sigma}^2 = \sigma^2 \left[1 + \frac{\kappa}{\sigma}\sqrt{\frac{2}{\pi \, \Delta t}}\right]
$$

where $\kappa$ = round-trip cost rate, $\Delta t$ = rebalance interval.

**In one sentence:** Pretend vol is higher than it is, which makes delta less sensitive
to spot moves, which means you trade less, which saves cost.

---

### Paper 8 — Whalley & Wilmott (1997): "Asymptotic Optimal Hedging"

**What we use:** the no-trade band concept.

**The result they derive:** under transaction costs, the optimal strategy isn't to
rehedge to exact delta. Instead, define a no-trade region around delta, and only
rehedge when you exit the band.

#### The bandwidth

$$
H \approx \left[\frac{3}{2} \cdot \frac{\kappa \cdot S \cdot \Gamma}{\rho_a}\right]^{1/3}
$$

where $\rho_a$ = risk aversion parameter.

**In one sentence:** When gamma is high or costs are high, accept more delta
mismatch — don't trade unless you're far out of band.

#### Our implementation

We use this as motivation for **8-hour rebalancing** (vs 1-hour). Tested grid:

| Cadence | Cost | Sharpe |
|---|---|---|
| 1h | $8.2K | 6.1 |
| **8h** | **$3.4K** | **7.7** |
| 24h | $1.8K | 6.8 |

8h is empirically Pareto-optimal. Saves 59% cost, *improves* Sharpe (less
noise-chasing).

---

### Paper 9 — Buehler et al. (2018): "Deep Hedging"

**What we use:** the framework of training a neural network to learn the optimal hedge,
end-to-end on the actual loss objective.

**The problem:** Black-Scholes delta assumes geometric Brownian motion, no jumps,
constant vol. Real BTC has fat tails, vol clustering, regime shifts.

**The solution:** use a neural net to learn delta directly from market features.

#### Network architecture

3-layer MLP, 64 hidden units. At each rebalance time $k$:

$$
\delta_k = F_\theta\left(\log(S_k/K),\, T_{\text{rem}},\, \sigma_k,\, \delta_{k-1},\, k_{\text{norm}}\right)
$$

The 5 features:
- $\log(S_k/K)$: moneyness
- $T_{\text{rem}}$: time left
- $\sigma_k$: current IV
- $\delta_{k-1}$: previous delta (for path dependence + cost smoothing)
- $k_{\text{norm}}$: normalized time index

#### Training objective: entropic risk

$$
\rho_\lambda(X) = \frac{1}{\lambda} \log \mathbb{E}\left[e^{-\lambda X}\right], \quad \lambda = 1
$$

**In one sentence:** Penalize losses *exponentially* — the network learns to be very
afraid of large losses, not just minimize expected loss.

This is equivalent to maximizing exponential utility (CARA).

#### Our deviation from the original paper

Buehler trained on synthetic GBM/Heston paths. We trained on **4,212 real BTC rolling
168-hour windows** from 2022-2023. This captures actual BTC dynamics (fat tails, vol
clusters, regime breaks) that synthetic models miss.

#### The big finding

Formal IS sweep over ensemble weight $w$ in $\delta = w \cdot \delta_{\text{DH}} + (1-w) \cdot \delta_{\text{BS}}$:

| $w$ | IS Sharpe | OOS Sharpe |
|---|---|---|
| 0.00 (pure BS) | **9.83** | **6.60** |
| 0.30 | 9.67 | 6.23 |
| 1.00 (pure DH) | 8.56 | 5.27 |

**$w^* = 0.00$**. Pure BS wins on Sharpe.

**Interpretation:** Deep hedge increases raw P&L (+9.3%) by being more aggressive in
tail-risk regimes, but also raises trade variance (win rate $88\% \to 74\%$). The
Buehler entropic objective ≠ Sharpe objective. Our finding is a clean empirical
demonstration of this gap.

---

## Part 6: Validation — Do We Trust the Numbers?

### Paper 10 — Politis & Romano (1994): "Stationary Bootstrap"

**What we use:** the stationary block bootstrap for confidence intervals.

**The problem:** plain bootstrap (random resampling) destroys time-series dependence.
Block bootstrap preserves it but is sensitive to fixed block size choice.

**The solution:** stationary bootstrap uses *random* block lengths drawn from a
geometric distribution. Robust to block size misspecification.

#### How it works

1. Pick a random starting trade index
2. Draw a block length from $\text{Geom}(p)$ where $1/p$ = mean block size
3. Take a contiguous block of that length (wrapping if needed)
4. Repeat until you have $n$ trades
5. Compute Sharpe on the resampled set
6. Repeat 10,000 times → empirical distribution of Sharpe

#### Our result

- Point Sharpe: 7.74
- Bootstrap mean: 7.7 ± 1.1
- **95% CI: [5.5, 9.7]**

Lower bound 5.5 is still 13× better than BTC's 0.42. The Sharpe is robust.

---

### Paper 11 — Corwin & Schultz (2012): "High-Low Spread Estimator"

**What we use:** estimate the bid-ask spread from high-low ranges (we don't have
top-of-book history).

#### The formula

$$
S = \frac{2(e^\alpha - 1)}{1 + e^\alpha}, \qquad \alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}
$$

where $\beta$ uses 2-day high-low ranges and $\gamma$ uses single-day ranges.

**In one sentence:** Convert high-low statistics into an effective spread estimate.
We use this to model BTC perpetual transaction costs realistically per hour.

---

## Part 7: Summary — How It All Fits Together

### Pipeline (per Friday)

```
1. Read option chain at 08:00 UTC (Carr-Wu integral on SVI surface)
   → SW²(7d), SW²(30d), ATM IV

2. Forecast realized vol from past 26 weeks (HAR-RV)
   → predicted RV²(7d)

3. Compute LRP = log(SW²) - log(forecast RV²)
   → composite signal score

4. Apply 5 gates (skip if any fails)
   → enter/skip decision

5. If enter: size position (Moreira-Muir vol target × VRP conviction × margin cap)
   → notional + n_straddles

6. Sell ATM straddle, hedge delta in perp every 8 hours
   → BS delta with Leland-modified σ, Whalley-Wilmott bandwidth in spirit

7. Monitor stop-loss (running RV > 1.5 × entry σ → exit)

8. Close at expiry (Friday +7d, 08:00 UTC)
   → realize P&L = θ × t + ½Γ × Δs² + V × Δσ - costs (Bakshi-Kapadia)
```

### Performance summary

| Metric | Value |
|---|---|
| OOS Sharpe (2024-2026) | 6.60 |
| Full-period Sharpe | 7.74 |
| Bootstrap 95% CI Sharpe | [5.5, 9.7] |
| Max drawdown | -1.2% |
| Annualized return | 22.7% (10× lev, $1M) |
| BTC buy-and-hold Sharpe | 0.42 |
| Win rate | 88% (42 trades) |

---

## Part 8: Reading List — In Order of Importance

If you read only one paper:
- **Carr & Wu (2009)** — defines what variance risk premium even is

If you read three:
- Carr & Wu (2009)
- **Corsi (2009)** — HAR-RV is the workhorse vol model
- **Buehler et al. (2018)** — modern deep-learning take on hedging

If you want the full intellectual chain:
1. Black-Scholes (1973) — option pricing foundation
2. Bakshi-Kapadia (2003) — Greek decomposition of delta-hedged P&L
3. Carr-Wu (2009) — model-free variance swap rate
4. Corsi (2009) — HAR-RV forecasting
5. Gatheral-Jacquier (2014) — SVI surface
6. Buehler et al. (2018) — deep hedging framework
7. Whalley-Wilmott (1997) — optimal hedge bandwidth
8. Leland (1985) — modified vol for transaction costs
9. Moreira-Muir (2017) — vol-managed sizing
10. Politis-Romano (1994) — stationary bootstrap CI
11. Corwin-Schultz (2012) — spread estimator

---

## Part 9: Common Misconceptions to Avoid

**"Deep hedging always beats BS."**
False. We showed the opposite for Sharpe-optimization. Deep hedge is better for
CVaR/tail-risk objectives. Pick objective first, then method.

**"Higher leverage = higher Sharpe."**
False. Sharpe is leverage-invariant for symmetric strategies. Leverage scales mean
*and* std proportionally. We use leverage to scale return — Sharpe stays the same.

**"VRP harvest is risk-free arbitrage."**
False. It's a risk premium — you bear short-vol, short-gamma risk in exchange for
the premium. Worst-case scenario is a black swan that the gates fail to catch
(none seen 2022-2026, but possible).

**"More trades = better."**
False. We're 19% selective. Adding more trades by lowering gates *worsened* Sharpe in
sweeps. Quality > quantity. Insurance company analogy: better to underwrite fewer,
clean policies than to take every applicant.

---

*Document path: `notebooks/STRATEGY_WALKTHROUGH.md`*
*Companion docs: `notebooks/MATH_AND_STRATEGY.md` (deeper math), `git/PRESENTATION_GUIDE.md` (meeting prep)*
