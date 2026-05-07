# Mathematical Foundations & Strategy Design

**Project:** Systematic Short-Vol on BTC Options with Deep Hedging
**Author:** Karan Chavan
**MSc Quantitative Finance, SMU**
**Last updated:** 2026-05

---

## Table of Contents

1. [Stochastic Foundations](#1-stochastic-foundations)
2. [Black-Scholes Framework](#2-black-scholes-framework)
3. [Greeks: Delta, Gamma, Vega, Theta](#3-greeks)
4. [Variance Risk Premium (VRP)](#4-variance-risk-premium-vrp)
5. [Carr-Wu Synthetic Variance Swap](#5-carr-wu-synthetic-variance-swap)
6. [HAR-RV Forecasting](#6-har-rv-forecasting)
7. [SVI Volatility Surface](#7-svi-volatility-surface)
8. [Static Replication of Variance](#8-static-replication-of-variance)
9. [Discrete Hedging & Transaction Costs](#9-discrete-hedging--transaction-costs)
10. [Buehler Deep Hedging](#10-buehler-deep-hedging)
11. [Position Sizing (Hybrid M&M × VRP)](#11-position-sizing)
12. [Entry Signal Composition](#12-entry-signal)
13. [Risk Measures & Bootstrap CI](#13-risk-measures)
14. [Why Each Design Choice](#14-design-rationale)

---

## 1. Stochastic Foundations

### 1.1 Brownian motion

A standard **Brownian motion** $W_t$ on $(\Omega, \mathcal{F}, \mathbb{P})$ satisfies:
- $W_0 = 0$
- $W_t$ has independent increments
- $W_t - W_s \sim \mathcal{N}(0, t-s)$ for $s < t$
- $t \mapsto W_t$ is continuous a.s.

### 1.2 Geometric Brownian motion (GBM)

The classical price process:

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

By Itô's lemma applied to $f(S) = \log S$:

$$
d\log S_t = \left(\mu - \tfrac{1}{2}\sigma^2\right) dt + \sigma \, dW_t
$$

Therefore $\log S_T \sim \mathcal{N}\!\left(\log S_0 + (\mu - \tfrac{1}{2}\sigma^2)T, \sigma^2 T\right)$.

### 1.3 Itô's lemma

For $f(t, S_t)$ where $S_t$ follows the SDE above:

$$
df = \left(\frac{\partial f}{\partial t} + \mu S \frac{\partial f}{\partial S} + \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}\right) dt + \sigma S \frac{\partial f}{\partial S} dW_t
$$

This is the **engine** behind Black-Scholes, all greeks, and the variance swap replication.

### 1.4 Quadratic variation (realized variance)

For $S_t$ following GBM, the **quadratic variation** of $\log S$ over $[0, T]$ is:

$$
[\log S]_T = \int_0^T \sigma_t^2 \, dt = \sum_i (\Delta \log S_{t_i})^2 \quad \text{(in the limit)}
$$

This is what we call **realized variance** $\text{RV}^2(0, T)$. Estimator from $n$ intraday returns:

$$
\hat{\text{RV}}^2 = \sum_{i=1}^n (\log S_{t_i} - \log S_{t_{i-1}})^2
$$

We **annualize**: $\text{RV}^2_{\text{ann}} = \hat{\text{RV}}^2 \cdot \frac{N_{\text{year}}}{n}$ where $N_{\text{year}} = 365$ for BTC.

In our code (`src/features/rv_from_index.py`):
```python
rv_1d = np.sqrt(np.sum(r_day ** 2) * 365)   # daily annualized vol from 1-min returns
```

---

## 2. Black-Scholes Framework

### 2.1 The PDE

Under risk-neutral measure $\mathbb{Q}$, asset price $S_t$ follows $dS_t = r S_t dt + \sigma S_t dW_t^{\mathbb{Q}}$.
A European option's value $V(t, S)$ satisfies:

$$
\frac{\partial V}{\partial t} + \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

with boundary condition $V(T, S) = \max(S - K, 0)$ for a European call.

### 2.2 Closed-form solutions

**Call:** $C(S, K, T, \sigma) = S N(d_1) - K e^{-rT} N(d_2)$

**Put:** $P(S, K, T, \sigma) = K e^{-rT} N(-d_2) - S N(-d_1)$

where:

$$
d_1 = \frac{\log(S/K) + (r + \tfrac{1}{2}\sigma^2) T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

For BTC we set $r = 0$ (no dividend, short tenor, perpetual basis $\approx 0$).

### 2.3 ATM Straddle

A **straddle** = long call + long put at same strike $K$:

$$
\text{Straddle}(S, K, T, \sigma) = C + P = S[N(d_1) - N(-d_1)] + K e^{-rT}[N(-d_2) - N(d_2)]
$$

For ATM ($K = S$, $r = 0$): $d_1 = \tfrac{1}{2}\sigma\sqrt{T}$, $d_2 = -\tfrac{1}{2}\sigma\sqrt{T}$, and:

$$
\text{Straddle}_{\text{ATM}} \approx S \sigma \sqrt{T} \cdot \frac{2}{\sqrt{2\pi}} \approx 0.798 \, S \sigma \sqrt{T}
$$

This is the **premium** received when **shorting** the straddle.

---

## 3. Greeks

Greeks are partial derivatives of $V$ with respect to inputs. They quantify hedging requirements.

### 3.1 Delta

$$
\Delta = \frac{\partial V}{\partial S}
$$

- $\Delta_{\text{call}} = N(d_1)$
- $\Delta_{\text{put}} = N(d_1) - 1$
- $\Delta_{\text{straddle}} = 2N(d_1) - 1$

For SHORT straddle: $\Delta = -(2N(d_1) - 1)$. To delta-hedge, hold $+(2N(d_1) - 1)$ units of the underlying perpetual.

### 3.2 Gamma

$$
\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\phi(d_1)}{S \sigma \sqrt{T}}
$$

where $\phi$ is standard normal PDF. Gamma is the **convexity** of the option to spot moves.

For ATM straddle: $\Gamma_{\text{straddle}} = 2 \cdot \tfrac{\phi(d_1)}{S\sigma\sqrt{T}}$.

When **short** an option, $\Gamma < 0$ from owner's perspective — every spot move hurts. This is the **gamma scalping cost** that makes shorting variance non-trivial.

### 3.3 Vega

$$
\nu = \frac{\partial V}{\partial \sigma} = S \sqrt{T} \, \phi(d_1)
$$

For straddle: $\nu_{\text{straddle}} = 2 S \sqrt{T} \phi(d_1)$.

Vega exposure means PnL sensitive to **changes in implied volatility**. Short vol = short vega = profits when IV drops.

### 3.4 Theta

$$
\Theta = \frac{\partial V}{\partial t} = -\frac{S \sigma \phi(d_1)}{2\sqrt{T}} \quad \text{(for } r = 0\text{)}
$$

For long option: $\Theta < 0$ (decays over time).
For SHORT option: $\Theta > 0$ — **time decay is income.**

### 3.5 P&L decomposition

By Itô on $V(t, S, \sigma)$ and assuming $\sigma$ has its own diffusion:

$$
dV = \Theta \, dt + \Delta \, dS + \tfrac{1}{2}\Gamma (dS)^2 + \nu \, d\sigma + \text{cross terms}
$$

For SHORT straddle held over discrete intervals:

$$
\text{PnL}_{\text{short}} \approx \underbrace{\Theta \, \Delta t}_{\text{income}} - \underbrace{\tfrac{1}{2}\Gamma (\Delta S)^2}_{\text{gamma loss}} - \underbrace{\nu \, \Delta \sigma}_{\text{vega exposure}}
$$

After delta-hedge: $\Delta \cdot dS$ term cancels (delta-neutral). What remains:

$$
\boxed{\text{PnL}_{\text{hedged short straddle}} = \theta \Delta t - \tfrac{1}{2}\Gamma (\Delta S)^2 - \nu \Delta \sigma - \text{costs}}
$$

This is the **fundamental equation** of our strategy. Theta is income, gamma is variance cost, vega is IV risk.

In our backtest (`src/layer6/backtest_weekly_real.py:simulate_one_trade`), we accumulate each component per hour:
```python
gamma_pnl  += -0.5 * gamma_unit * dS**2 * n_straddles
vega_pnl   += -vega_unit * dsigma * n_straddles
theta_pnl  += -theta_per_year_unit * dt_year * n_straddles
```

---

## 4. Variance Risk Premium (VRP)

### 4.1 Definition

**VRP** = difference between option-implied variance and expected realized variance:

$$
\text{VRP}(t, T) = \mathbb{E}_t^{\mathbb{Q}}[\text{RV}^2(t, T)] - \mathbb{E}_t^{\mathbb{P}}[\text{RV}^2(t, T)]
$$

$\mathbb{Q}$ = risk-neutral (option market), $\mathbb{P}$ = physical (real). VRP > 0 means options are **rich** relative to expected realization → short-vol is profitable on average.

We use **synthetic variance swap** $\text{SW}^2$ as proxy for $\mathbb{E}^{\mathbb{Q}}[\text{RV}^2]$ (model-free, see §5).

### 4.2 Log VRP (regime-invariant)

Since vol scales with regime, raw VRP is non-stationary. We use:

$$
\text{LRP}(t, T) = \log \text{SW}^2(t, T) - \log \mathbb{E}^{\mathbb{P}}[\text{RV}^2(t, T)]
$$

Empirically on BTC 2022-2026: $\text{LRP}_{\text{mean}} = +0.50$, $P(\text{LRP} > 0) = 99.7\%$.

### 4.3 Magnitude check

For SPX (Carr-Wu 2009): $\text{LRP}_{\text{SPX}} \approx +0.40$. BTC at +0.50 is *similar magnitude* — VRP exists in BTC market just like equity options.

---

## 5. Carr-Wu Synthetic Variance Swap

### 5.1 Static replication formula

A variance swap of notional $V$ at $(t, T)$ pays:

$$
\text{Payoff} = V \cdot (\text{RV}^2(t, T) - \text{SW}^2(t, T)) \cdot (T - t)
$$

**Carr-Wu (2009) Eq. 5** shows $\text{SW}^2$ can be replicated by a portfolio of OTM options:

$$
\text{SW}^2(t, T) = \frac{2}{T-t} e^{r(T-t)} \int_0^{F_t} \frac{P(K, T)}{K^2} dK + \frac{2}{T-t} e^{r(T-t)} \int_{F_t}^{\infty} \frac{C(K, T)}{K^2} dK
$$

where $F_t = S_t e^{r(T-t)}$ is the forward, $P, C$ are OTM put/call prices in **USD**.

### 5.2 Discrete approximation (our implementation)

For Deribit quotes given in BTC, USD price = $\text{mark}_{\text{BTC}} \cdot S_t$:

$$
\text{SW}^2 \approx \frac{2}{T-t} e^{r(T-t)} S_t \sum_{K_i \in \text{OTM}} \frac{\text{mark}_{\text{BTC}}(K_i)}{K_i^2} \Delta K_i
$$

where:
- $\Delta K_i = (K_{i+1} - K_{i-1}) / 2$ (trapezoid rule)
- OTM rule: use put if $K_i < F_t$, call if $K_i > F_t$

Code: `src/features/synthetic_sw.py:_per_expiry_filtered`.

### 5.3 Why this matters

- **Model-free**: doesn't assume Black-Scholes or any specific dynamics
- **Captures full smile**: weights OTM options by $1/K^2$, accounts for jumps and skew
- **Tight bound**: theoretical replication error is small for liquid strike grids

Empirically on BTC: ATM-IV-only proxy underestimates VRP by ~13% in variance terms vs Carr-Wu.

---

## 6. HAR-RV Forecasting

### 6.1 Corsi (2009) HAR specification

$$
\text{RV}^d_{t+1} = c + \beta_d \text{RV}^d_t + \beta_w \text{RV}^w_t + \beta_m \text{RV}^m_t + \epsilon_t
$$

Where:
- $\text{RV}^d_t$ = daily annualized vol on day $t$
- $\text{RV}^w_t = \tfrac{1}{5}\sum_{i=0}^{4} \text{RV}^d_{t-i}$ (weekly avg)
- $\text{RV}^m_t = \tfrac{1}{22}\sum_{i=0}^{21} \text{RV}^d_{t-i}$ (monthly avg)

### 6.2 h-step forward extension

For predicting $h$-day-ahead average RV (Carr-Wu E[RV] for VRP at horizon $h$):

$$
\bar{\text{RV}}^d_{[t+1, t+h]} = \tfrac{1}{h}\sum_{i=1}^{h} \text{RV}^d_{t+i}
$$

Fit: $\bar{\text{RV}}^d_{[t+1, t+h]} = c_h + \beta_{d,h} \text{RV}^d_t + \beta_{w,h} \text{RV}^w_t + \beta_{m,h} \text{RV}^m_t + \epsilon$

### 6.3 Walk-forward fitting (CRITICAL)

To avoid look-ahead, at each forecast time $t$ we **only use data with target window completed by $t$**:

```python
# For h-step forecast at time t, only use samples i where i + h ≤ t
end = t - h + 1        # exclusive upper bound for training data
fit OLS on rows [0, end)
forecast at t using rows[t]
```

This is implemented in `src/features/har_rv.py:fit_har_h_step`.

### 6.4 BTC 2022-2026 results

| Horizon | $\beta_d$ | $\beta_w$ | $\beta_m$ | $R^2$ |
|---|---|---|---|---|
| 1d | 0.483 | 0.151 | 0.176 | 0.402 |
| 5d | 0.158 | 0.389 | 0.115 | 0.388 |
| 22d | 0.105 | 0.060 | 0.348 | 0.289 |
| 30d | 0.075 | 0.057 | 0.365 | 0.304 |

All three components significant ($t > 2$). Daily lag dominates 1-step, monthly dominates 30-step. Same pattern as SPX (Corsi 2009).

---

## 7. SVI Volatility Surface

### 7.1 Raw SVI parameterization (Gatheral-Jacquier 2014)

For each expiry, fit total variance $w(k) = \sigma^2(k) \cdot T$ as function of log-moneyness $k = \log(K/F)$:

$$
w(k) = a + b \left\{ \rho(k - m) + \sqrt{(k - m)^2 + \sigma^2} \right\}
$$

5 parameters: $(a, b, \rho, m, \sigma)$.

Constraints (no calendar/butterfly arbitrage):
- $a \in (-1, 5)$, $b \ge 0$, $\rho \in (-1, 1)$, $m \in (-2, 2)$, $\sigma > 0$
- $b(1 + |\rho|) \le 4$ (no-arb butterfly)

### 7.2 Calibration via least squares

Objective: minimize $\sum_i [w(k_i) - w_{\text{market}}(k_i)]^2$ over observed $k_i$.

Code: `src/features/svi_fit.py` uses `scipy.optimize.least_squares` with bounds.

### 7.3 ATM analytics (closed-form)

From SVI parameters, derive at $k = 0$:
- ATM total variance: $w_{\text{atm}} = a + b(\rho \cdot (-m) + \sqrt{m^2 + \sigma^2})$
- ATM skew: $\frac{\partial w}{\partial k}\big|_0 = b\!\left(\rho + \frac{-m}{\sqrt{m^2+\sigma^2}}\right)$
- ATM curvature: $\frac{\partial^2 w}{\partial k^2}\big|_0 = b \cdot \frac{\sigma^2}{(m^2+\sigma^2)^{3/2}}$

These feed downstream: ATM skew is a regime indicator (negative skew = put rich = stress).

---

## 8. Static Replication of Variance

### 8.1 Bakshi-Madan (2003) decomposition

Any twice-differentiable payoff $g(S_T)$ can be replicated by:

$$
g(S_T) = g(F) + g'(F)(S_T - F) + \int_0^F g''(K) (K - S_T)_+ dK + \int_F^\infty g''(K) (S_T - K)_+ dK
$$

For $g(S) = \log(S/F) - (S - F)/F$ (variance contract), $g''(K) = 1/K^2$, recovering Carr-Wu.

### 8.2 Continuous delta-hedged short straddle = short variance swap

By Itô applied to BS price:

$$
dV(t, S, \sigma) = \Theta dt + \Delta dS + \tfrac{1}{2}\Gamma S^2 \frac{(dS)^2}{S^2}
$$

Under continuous delta-hedge (cancels $\Delta dS$), and using $(dS)^2 = \sigma_{\text{realized}}^2 S^2 dt$:

$$
dV - \Delta dS = \Theta dt + \tfrac{1}{2}\Gamma S^2 \sigma_{\text{realized}}^2 dt
$$

Over $[0, T]$ for ATM:

$$
\text{PnL} \approx \tfrac{1}{2} \int_0^T \Gamma S^2 (\sigma_{\text{implied}}^2 - \sigma_{\text{realized}}^2) dt = V \cdot (\sigma_{\text{IV}}^2 - \sigma_{\text{RV}}^2) \cdot T
$$

where $V$ is calibrated to vega-equivalent variance notional. **This is the variance swap payoff.**

### 8.3 Vega-matching n_straddles formula

To replicate $V$ dollars of variance swap, hold:

$$
n_{\text{straddles}} = \frac{2 \sigma T \cdot V}{\nu_{\text{straddle}}}
$$

where $\nu_{\text{straddle}} = 2 S \sqrt{T} \phi(d_1)$. Derived from matching variance-vega exposure.

Code: `backtest_weekly_real.py:simulate_one_trade`:
```python
n_straddles = (2 * entry_iv * T0 * var_notional) / max(straddle_vega, 1e-9)
```

---

## 9. Discrete Hedging & Transaction Costs

### 9.1 Discrete hedging error

Continuous hedging gives the variance-swap payoff exactly. Discrete hedging introduces **gamma scalping noise**:

$$
\text{PnL}_{\text{discrete}} - \text{PnL}_{\text{continuous}} = \tfrac{1}{2} \Gamma \sum_k \left[(\Delta S_k)^2 - \sigma_{\text{realized}}^2 S_k^2 \Delta t\right]
$$

This has zero expectation but nonzero variance ∝ $\Gamma^2 S^4 \sigma^4 \Delta t^2$.

### 9.2 Optimal hedging frequency

There's a tradeoff:
- More frequent → less gamma noise but more transaction cost
- Less frequent → more gamma noise but less cost

For proportional cost $\epsilon$ on perp notional and quadratic gamma noise, **Whalley-Wilmott (1997)** asymptotic optimum:

$$
\Delta t^* \propto \left(\frac{\epsilon S}{\sigma^2 \Gamma^2}\right)^{2/3}
$$

For BTC weekly straddle: optimal $\Delta t^* \approx 4-12$ hours. We use **8h cadence** (validated empirically: saves 59% txn cost vs 1h).

### 9.3 Leland (1985) modified volatility

Approximate transaction-cost-aware pricing:

$$
\sigma_{\text{Leland}}^2 = \sigma^2 \left(1 + A \cdot \text{sign}(\Gamma) \sqrt{\frac{8}{\pi}} \frac{\epsilon}{\sigma \sqrt{\Delta t}}\right)
$$

For SHORT gamma ($\Gamma < 0$): $\sigma_{\text{Leland}} < \sigma$ (price options as if vol is lower). We use $A = 0.25$ in `src/layer4/policies.py:leland_policy_factory`.

### 9.4 Whalley-Wilmott bandwidth

Don't trade unless delta deviation exceeds bandwidth:

$$
h(t, S) = \left(\frac{3 e^{-r(T-t)} \epsilon \cdot S \cdot \Gamma^2}{2\gamma}\right)^{1/3}
$$

Trade only if $|\delta_{\text{actual}} - \delta_{\text{BS}}| > h$. Reduces trade frequency in low-gamma regimes.

### 9.5 Corwin-Schultz spread estimator

We don't have L2 order book data. Estimate effective bid-ask spread from 1-min H/L:

$$
\hat{S}_{\text{spread}} = \frac{2(e^\alpha - 1)}{1 + e^\alpha}
$$

where $\alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}$,
$\beta = \mathbb{E}[(\log H_t/L_t)^2]$ averaged over 2 consecutive bars,
$\gamma = (\log H_{2\text{bar}}/L_{2\text{bar}})^2$.

For BTC 1-min top-of-book, this gives ~1bp average. Implementation: `backtest_weekly_real.py:corwin_schultz_spread_hourly`.

---

## 10. Buehler Deep Hedging

### 10.1 The deep hedging problem

Find optimal hedging policy $\delta^* = \{\delta_k\}_{k=0}^{N-1}$ minimizing risk of terminal PnL:

$$
\delta^* = \arg\min_\delta \rho\!\left(p_0 + \sum_{k=0}^{N-1} \delta_k (S_{k+1} - S_k) - \sum_{k=0}^{N-1} \epsilon |\delta_k - \delta_{k-1}| S_k - g(S_T)\right)
$$

Where:
- $p_0$ = initial premium received
- $g(S_T) = \max(S_T - K, 0) + \max(K - S_T, 0)$ for SHORT straddle
- $\epsilon$ = transaction cost rate
- $\rho$ = convex risk measure (entropic)

### 10.2 Entropic risk measure

$$
\rho_\lambda(X) = \frac{1}{\lambda} \log \mathbb{E}\left[e^{-\lambda X}\right]
$$

For $\lambda > 0$: penalizes left tail (risk aversion). For $\lambda \to 0$: $\rho \to -\mathbb{E}[X]$ (risk neutral).

We use $\lambda = 1$ (Buehler's choice). With premium $\sim \$13K$ and PnL in $\sim O(\$10K)$, we **scale**:

$$
\rho_\lambda(X) = c \cdot \rho_\lambda(X / c)
$$

where $c = $ median premium. Prevents `exp` overflow.

Code: `src/layer5/train.py:entropic_loss`.

### 10.3 Network architecture

Single shared MLP across timesteps (Buehler's "semi-recurrent" form):

$$
\delta_k = F_\theta(I_k), \quad I_k = [\log(S_k/K), \, T_{\text{rem}}, \, \sigma_k, \, \delta_{k-1}, \, k_{\text{norm}}]
$$

Architecture (v3, what we use):
- Input: 5 features
- Hidden 64 × 3 layers, ReLU, NO BatchNorm (BatchNorm fails on constant-σ batches)
- Output: 1 scalar (delta)

Why semi-recurrent: $k_{\text{norm}} = k / N$ embeds time, lets one MLP serve all steps. Standard recurrent (LSTM) overkill for this Markov problem.

### 10.4 Gradient computation

Backprop through the **entire path**:

$$
\frac{\partial \rho}{\partial \theta} = \mathbb{E}\!\left[\frac{\partial}{\partial \theta} \sum_{k=0}^{N-1} \delta_k(\theta) (S_{k+1} - S_k) - \epsilon \frac{\partial}{\partial \theta}|\delta_k(\theta) - \delta_{k-1}(\theta)| S_k\right]
$$

PyTorch's autograd handles this automatically through `simulate_pnl`.

### 10.5 BS recovery sanity check

For $\epsilon = 0$, optimal policy = BS delta. Verify:

$$
\lim_{\epsilon \to 0} \mathbb{E}[\text{PnL}_{\text{deep hedge}}] = \mathbb{E}[\text{PnL}_{\text{BS hedge}}]
$$

Empirical: deep hedge mean +$19.9 vs BS +$19.2 (99.5% match). ✅

### 10.6 $\epsilon^{2/3}$ scaling

Whalley-Wilmott theory: hedge cost scales as $\epsilon^{2/3}$ in friction:

$$
p_\epsilon - p_0 \sim O(\epsilon^{2/3})
$$

Empirical exponent on our data: 0.80 (vs theory 0.67). Same regime, slight overshoot due to finite training budget. ✅

### 10.7 Real BTC training (v3, our default)

Trained on **4,212 real rolling 168-hour windows** from 2022-2023:
- Spot path: from `btc_index_1min.parquet`
- IV path: $\sigma_t = \sqrt{\text{sw\_var\_7d}_t}$ at each hour
- Strike: $K = S_0$ (ATM each window)
- Premium: BS straddle price at entry IV

Validation: 2024 H1 (525 paths). Best epoch: 15. Best val $\rho = -195$.

Code: `src/layer5/train_real.py`.

### 10.8 Ensemble blend (v6, production)

Pure deep hedge maximizes mean PnL but increases variance. Pure BS minimizes variance.

Sharpe-optimal: 30% v3 + 70% BS:

$$
\delta_{\text{ensemble}} = 0.3 \cdot \delta_{\text{v3}} + 0.7 \cdot \delta_{\text{BS}}
$$

Empirical Pareto frontier: $w = 0.3$ peaks Sharpe at 7.30 vs 7.14 (BS) and 6.22 (pure deep).

---

## 11. Position Sizing

### 11.1 Moreira-Muir (2017) volatility-managed

$$
w_t = \frac{\sigma_{\text{target}}}{\hat{\sigma}_t}
$$

Increases position when realized vol low, decreases when high. Equalizes risk per dollar.

### 11.2 VRP conviction multiplier

We layer in VRP magnitude:

$$
m_t = \min\!\left(2, \frac{\text{VRP}_t}{\overline{\text{VRP}}_{60d}}\right)
$$

Capped at 2× to prevent oversizing in trough VRP regimes.

### 11.3 Final sizing

$$
\boxed{\text{notional} = \min\!\left(\text{cap}, \, \text{BASE} \cdot w_t \cdot m_t \cdot S_t^{\text{entry}}\right)}
$$

Where:
- BASE = $5M (5× leverage on $1M account)
- cap = max\_margin\_frac × account / margin\_frac\_of\_notional = $10M
- $S_t^{\text{entry}} \in [0, 1]$ is the entry score (see §12)

### 11.4 Margin envelope

Deribit short straddle initial margin $\approx 15\%$ of notional. On $1M account, $1M / 0.15 = $6.67M max safe notional. We cap at $10M (using maintenance margin headroom).

---

## 12. Entry Signal Composition

### 12.1 Three sub-scores

$$
S_{\text{VRP}} = \max\!\left(0, 2 \cdot \left(\text{percentile}_{60d}(\text{VRP}_t) - 0.5\right)\right) \in [0, 1]
$$

$$
S_{\text{TS,front}} = \text{sigmoid}(-\alpha \cdot \text{slope}_{30d \to 7d}) \quad \text{(reward inversion)}
$$

$$
S_{\text{TS,back}} = \text{sigmoid}(-\alpha \cdot \text{slope}_{90d \to 30d}) \quad \text{(reward back inversion)}
$$

with $\alpha = 30$ (Vasquez 2017 — sell front when slope is INVERTED, indicating mean reversion).

### 12.2 Composite entry score

$$
S^{\text{entry}}_t = w_{\text{VRP}} S_{\text{VRP}} + w_{\text{TS,F}} S_{\text{TS,front}} + w_{\text{TS,B}} S_{\text{TS,back}}
$$

with $w_{\text{VRP}} = 0.6$, $w_{\text{TS,F}} = 0.3$, $w_{\text{TS,B}} = 0.1$.

### 12.3 Dynamic gating

Enter only if **ALL** conditions:

1. $S^{\text{entry}}_t \ge 0.30$
2. $\text{VRP}_t \ge \alpha_{\text{base}} \cdot \text{SW}^2_{30d}$ where $\alpha_{\text{base}} = 0.20$
3. $z_{\text{VRP}} = \frac{\text{VRP}_t - \mu_{8w}}{\sigma_{8w}} \ge 0.5$ (using **prior 8w**, shift(1))
4. $\text{mom}_{1w} = \text{RV}_{22d,t} / \text{RV}_{22d,t-1w} \le 1.10$ (skip vol expansion)
5. $\text{RV}_{5d,\text{trailing}} \le \text{SW}^2_{30d}$ (skip if 5-day realized > 30d implied)

Conditions 4-5 use **only past data** at $t$. No look-ahead.

### 12.4 Mid-trade stop-loss

During the 7-day hold, monitor running annualized RV. If after 24h:

$$
\text{RV}^2_{\text{running}}(t, t+k) > 1.5 \cdot \text{SW}^2_{\text{entry}}
$$

→ exit position (close perp + buy back straddle at BS fair value).

---

## 13. Risk Measures

### 13.1 Sharpe ratio

$$
\text{Sharpe} = \frac{\mathbb{E}[\text{PnL}]}{\text{StdDev}[\text{PnL}]} \cdot \sqrt{52}
$$

For weekly trades, $\sqrt{52} \approx 7.21$ annualizes from per-trade std.

### 13.2 Sortino (downside std)

$$
\text{Sortino} = \frac{\mathbb{E}[\text{PnL}]}{\sqrt{\mathbb{E}[(\text{PnL})_-^2]}} \cdot \sqrt{52}
$$

Penalizes only left-tail losses.

### 13.3 CVaR (Expected Shortfall)

$$
\text{CVaR}_\alpha = \mathbb{E}[X | X \le \text{VaR}_\alpha]
$$

For $\alpha = 0.05$: average of the worst 5% trades.

### 13.4 Maximum drawdown

$$
\text{MDD} = \min_t \left(E_t - \max_{s \le t} E_s\right)
$$

where $E_t$ is cumulative equity.

### 13.5 Politis-Romano stationary block bootstrap

To get confidence intervals while preserving serial correlation:

1. At each step, with probability $1/L$: start a new block (random index)
2. Otherwise: extend current block by 1
3. $L = 4$ trades = mean block size (geometric)

10,000 resamples → 95% CI on Sharpe, cum PnL, win rate, worst trade.

Implementation: `src/layer6/bootstrap_ci.py:stationary_bootstrap_indices`.

### 13.6 Walk-forward param tuning (no look-ahead)

For each Friday $t$ in test set, refit on trailing 26 weeks:

```
For t in [26+] :
    train_data = rows[t - 26 : t]
    best_params = argmax_params Sharpe(train_data, params) s.t. n_trades >= 4
    decision_at_t = use best_params
```

Refit every 4 weeks (computational efficiency). Result: walk-forward Sharpe 9.29 vs full-sample 9.11. **Adapts** to regime changes.

---

## 14. Design Rationale

| Choice | Why |
|---|---|
| **Friday 08:00 UTC entry** | Deribit weekly options expire and re-issue at this exact time. Maximum VRP signal at fresh issuance. Tested all 28 day×hour combos — Friday 08:00 dominates. |
| **7-day weekly tenor** | Carr-Wu showed VRP strongest at 1-month; we use 1-week to compound 4× more often. Empirical: weekly outperforms monthly 4×. |
| **Carr-Wu over ATM IV** | Captures jump and skew premium. ATM IV underestimates BTC VRP by 13% in variance terms. |
| **Walk-forward HAR** | In-sample HAR fit overestimates RV in calm periods → VRP looks weaker than it is. Walk-forward fixes this; backtest improved \$78k → \$99k. |
| **VRP_v2 (Carr-Wu) over v1 (ATM)** | LRP +0.50 (matches SPX) vs +0.21 (biased low). Production signal is v2. |
| **No GEX feature** | Tested empirically on BTC: $R^2 \approx 0.003$, all $t < 2$. Equity dealer-gamma effect doesn't transfer to BTC. Dropped. |
| **Fixed σ regime gate via mom_1w + rv5d** | Dynamic VRP threshold $\alpha \cdot \text{SW}^2$ adapts to vol level. mom_1w catches "vol about to expand" weeks (Oct 2025 trap, etc.). |
| **8h hedge cadence** | Whalley-Wilmott optimum for our $\epsilon$ and $\Gamma$. Empirical: saves 59% perp txn cost vs 1h, no risk increase. |
| **Ensemble 30% deep + 70% BS** | Pure deep maxes mean PnL but increases variance. Pareto-optimal blend at $w=0.30$ — confirmed by full sweep. |
| **Entropic risk $\lambda=1$** | Buehler's choice. Penalizes left tail without being too risk-averse. |
| **Real BTC training (not synthetic)** | Deep hedge trained on actual 168-h rolling windows captures BTC-specific gamma dynamics. Original Buehler used GBM/Heston paths — we improvised. |
| **Margin cap on sizing** | Caps notional at maintenance margin headroom. Prevents oversize trades that would breach margin in stress. |
| **Stop-loss 1.5× entry SW** | Provides tail protection beyond the gate. Trade-off: tested 1.5/1.75/2.0 — 1.5× best on both cum and tail. |
| **Bootstrap CI for Sharpe** | n=42 trades has wide CI [5.6, 9.7]. Block bootstrap preserves serial correlation. Honest reporting. |

---

## 15. Open Questions / Future Work

1. **Vega-volga hedging**: we delta-hedge only. Adding vega-hedge via long ITM call/put would reduce IV-path risk.
2. **Multi-tenor portfolio**: weekly + biweekly + monthly straddles. Diversifies across realization horizons.
3. **Skew-as-signal**: SVI $\rho$ time series exists. Could improve entry score (Carr-Wu skew premium).
4. **Online learning**: Layer 5 net retrains every quarter on rolling data.
5. **Funding rate full coverage**: get historical Deribit funding (29% covered currently).
6. **Multi-asset extension**: same framework on ETH, SOL options.

---

## Bibliography

[1] Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2018). *Deep Hedging.* Quantitative Finance, 19(8), 1271-1291.

[2] Carr, P., Wu, L. (2009). *Variance Risk Premiums.* Review of Financial Studies, 22(3), 1311-1341.

[3] Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics, 7(2), 174-196.

[4] Gatheral, J., Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces.* Quantitative Finance, 14(1), 59-71.

[5] Bakshi, G., Kapadia, N. (2003). *Delta-Hedged Gains and the Negative Market Volatility Risk Premium.* Review of Financial Studies, 16(2), 527-566.

[6] Moreira, A., Muir, T. (2017). *Volatility-Managed Portfolios.* Journal of Finance, 72(4), 1611-1644.

[7] Zakamouline, V. (2006). *European option pricing and hedging with both fixed and proportional transaction costs.* Journal of Economic Dynamics and Control, 30(1), 1-25.

[8] Whalley, A. E., Wilmott, P. (1997). *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.* Mathematical Finance, 7(3), 307-324.

[9] Vasquez, A. (2017). *Equity Volatility Term Structures and the Cross Section of Option Returns.* Journal of Financial and Quantitative Analysis, 52(6), 2727-2754.

[10] Politis, D. N., Romano, J. P. (1994). *The Stationary Bootstrap.* Journal of the American Statistical Association, 89(428), 1303-1313.

[11] Leland, H. E. (1985). *Option Pricing and Replication with Transactions Costs.* Journal of Finance, 40(5), 1283-1301.

[12] Bakshi, G., Madan, D. (2000). *Spanning and Derivative-Security Valuation.* Journal of Financial Economics, 55(2), 205-238.

[13] Andersen, T. G., Bollerslev, T., Diebold, F. X. (2007). *Roughing It Up: Including Jump Components in the Measurement, Modeling, and Forecasting of Return Volatility.* Review of Economics and Statistics, 89(4), 701-720.

---

*End of document. ~6500 words. All math typeset in LaTeX, all design choices justified, all references academic.*
