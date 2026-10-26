"""Fit a damage-ratio curve per segment from observed county losses.

For one segment, each observation is a county in an event:

    modelled_i = tiv_i x sum_b(f_ib x DR_b)

Parameters are ``theta_b = log DR_b``. The objective is

    J = mean_i(w_i x r_i^2) + alpha x mean_e(R_e^2) + lam x mean_b((theta_b - prior_b)^2)

    r_i = log(modelled_i + eps_i) - log(actual_i + eps_i)      county error
    R_e = log(sum modelled) - log(sum actual) over event e     event-total error

``eps_i`` is a 0.01% damage-ratio floor so a reported zero is a penalty, not
infinity. A below-materiality row (``upper_bound`` set) is interval-censored:
``r_i = max(0, log(modelled_i + eps_i) - log(upper_i + eps_i))``. PCS spreads
token amounts (often round $5k-$40k) over barely touched counties; read as
exact they imply absurdly low damage ratios and drag the low-wind bands down. The ridge toward the prior (Form V-1 KCC by default) keeps bands
with little data in place. After every step DR is projected onto a curve
that never falls as wind rises. Pure Python: about a dozen parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DR_MIN = 1e-5
DR_MAX = 1.0
EPS_DR = 1e-4


@dataclass(slots=True)
class Observation:
    event: str
    geography_id: str
    tiv: float
    bins: list[tuple[int, float]]  # (band index, area fraction)
    actual: float
    # Set for below-materiality rows: the true loss is only known to lie in
    # [0, upper_bound], so the fit is penalised only above it.
    upper_bound: float | None = None


@dataclass(slots=True)
class FitOptions:
    # ASSUMPTION: equal say for county shape and event totals, light ridge.
    alpha_event: float = 1.0
    ridge: float = 0.05
    max_iter: int = 3000
    tol: float = 1e-10
    # Penalty on the second difference of log DR (0 = off).
    smooth: float = 0.0
    # (band index, DR) pairs held at a value the data here cannot identify.
    fixed: tuple[tuple[int, float], ...] = ()
    # Bands above the last one holding at least this share of the exposure
    # follow the prior's shape, scaled at that band (0 = off). Without it the
    # smoothness penalty extends the fitted log slope into bands with no data.
    tail_min_weight: float = 0.0


@dataclass(slots=True)
class FitResult:
    curve: list[float]
    objective: float
    iterations: int
    n_obs: int
    events: list[str]
    county_sigma: float | None
    event_errors: dict[str, float]
    band_weight: list[float]
    oos_county_sigma: float | None = None
    oos_event_errors: dict[str, float] = field(default_factory=dict)
    # Curve fitted without each event, for scoring that event unseen.
    oos_curves: dict[str, list[float]] = field(default_factory=dict)


def _pav_non_decreasing(values: list[float]) -> list[float]:
    """Pool-adjacent-violators: closest non-decreasing sequence (least squares)."""
    blocks: list[list[float]] = []  # [sum, count]
    for v in values:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out: list[float] = []
    for s, c in blocks:
        out.extend([s / c] * int(c))
    return out


def _project(theta: list[float], fixed: tuple[tuple[int, float], ...] = ()) -> list[float]:
    lo, hi = math.log(DR_MIN), math.log(DR_MAX)
    out = [min(hi, max(lo, t)) for t in _pav_non_decreasing(theta)]
    if fixed:
        for b, v in fixed:
            out[b] = math.log(min(DR_MAX, max(DR_MIN, v)))
        out = [min(hi, max(lo, t)) for t in _pav_non_decreasing(out)]
        for b, v in fixed:
            out[b] = math.log(min(DR_MAX, max(DR_MIN, v)))
    return out


def _weights(obs: list[Observation]) -> list[float]:
    # ASSUMPTION: sqrt(actual) weighting. Large counties count for more without
    # one Miami-Dade drowning out every other county.
    raw = [math.sqrt(max(o.actual, 0.0) + EPS_DR * o.tiv) for o in obs]
    mean = sum(raw) / len(raw) if raw else 1.0
    return [r / mean if mean > 0 else 1.0 for r in raw]


def _evaluate(
    theta: list[float],
    obs: list[Observation],
    weights: list[float],
    prior: list[float],
    opts: FitOptions,
    want_grad: bool,
) -> tuple[float, list[float]]:
    nb = len(theta)
    dr = [math.exp(t) for t in theta]
    grad = [0.0] * nb
    j_county = 0.0
    ev_model: dict[str, float] = {}
    ev_actual: dict[str, float] = {}
    ev_dm: dict[str, list[float]] = {}
    for o, w in zip(obs, weights):
        m = o.tiv * sum(f * dr[b] for b, f in o.bins)
        eps = EPS_DR * o.tiv
        if o.upper_bound is None:
            r = math.log(m + eps) - math.log(o.actual + eps)
        else:
            r = max(0.0, math.log(m + eps) - math.log(o.upper_bound + eps))
        j_county += w * r * r
        ev_model[o.event] = ev_model.get(o.event, 0.0) + m
        ev_actual[o.event] = ev_actual.get(o.event, 0.0) + o.actual
        if want_grad:
            dm = ev_dm.setdefault(o.event, [0.0] * nb)
            scale = 2.0 * w * r / (m + eps)
            for b, f in o.bins:
                part = o.tiv * f * dr[b]
                grad[b] += scale * part
                dm[b] += part
    n = max(1, len(obs))
    j = j_county / n
    if want_grad:
        grad = [g / n for g in grad]
    events = [e for e in ev_model if ev_actual[e] > 0 and ev_model[e] > 0]
    if events and opts.alpha_event > 0:
        ne = len(events)
        for e in events:
            big_r = math.log(ev_model[e]) - math.log(ev_actual[e])
            j += opts.alpha_event * big_r * big_r / ne
            if want_grad:
                k = 2.0 * opts.alpha_event * big_r / (ne * ev_model[e])
                for b, part in enumerate(ev_dm[e]):
                    grad[b] += k * part
    for b in range(nb):
        d = theta[b] - prior[b]
        j += opts.ridge * d * d / nb
        if want_grad:
            grad[b] += 2.0 * opts.ridge * d / nb
    if opts.smooth > 0 and nb >= 3:
        # Second difference of log DR: the log curve should bend, not step.
        m = nb - 2
        for b in range(1, nb - 1):
            c2 = theta[b + 1] - 2.0 * theta[b] + theta[b - 1]
            j += opts.smooth * c2 * c2 / m
            if want_grad:
                g = 2.0 * opts.smooth * c2 / m
                grad[b + 1] += g
                grad[b] -= 2.0 * g
                grad[b - 1] += g
    return j, grad


def fit_curve(
    obs: list[Observation],
    prior_curve: list[float],
    opts: FitOptions | None = None,
) -> FitResult:
    opts = opts or FitOptions()
    prior = [math.log(min(DR_MAX, max(DR_MIN, v))) for v in prior_curve]
    nb = len(prior)
    band_weight = [0.0] * nb
    for o in obs:
        for b, f in o.bins:
            band_weight[b] += f * o.tiv
    total_w = sum(band_weight) or 1.0
    band_weight = [w / total_w for w in band_weight]
    if not obs:
        return FitResult(list(prior_curve), 0.0, 0, 0, [], None, {}, band_weight)

    weights = _weights(obs)
    theta = _project(list(prior), opts.fixed)
    j, grad = _evaluate(theta, obs, weights, prior, opts, True)
    step = 1.0
    it = 0
    for it in range(1, opts.max_iter + 1):
        while True:
            cand = _project([t - step * g for t, g in zip(theta, grad)], opts.fixed)
            j_new, _ = _evaluate(cand, obs, weights, prior, opts, False)
            decrease = sum(g * (t - c) for g, t, c in zip(grad, theta, cand))
            if j_new <= j - 1e-4 * decrease or step < 1e-12:
                break
            step *= 0.5
        moved = max(abs(c - t) for c, t in zip(cand, theta))
        improvement = j - j_new
        theta = cand
        j, grad = _evaluate(theta, obs, weights, prior, opts, True)
        step = min(step * 2.0, 64.0)
        if moved < 1e-9 or 0 <= improvement < opts.tol:
            break

    if opts.tail_min_weight > 0:
        theta = _prior_tail(theta, prior, band_weight, opts.tail_min_weight, opts.fixed)
    curve = [math.exp(t) for t in theta]
    county_res, event_err = residuals(obs, curve)
    return FitResult(
        curve=curve,
        objective=j,
        iterations=it,
        n_obs=len(obs),
        events=sorted({o.event for o in obs}),
        county_sigma=_rms(county_res),
        event_errors=event_err,
        band_weight=band_weight,
    )


def _prior_tail(
    theta: list[float],
    prior: list[float],
    band_weight: list[float],
    min_weight: float,
    fixed: tuple[tuple[int, float], ...],
) -> list[float]:
    supported = [b for b, w in enumerate(band_weight) if w >= min_weight]
    if not supported:
        return theta
    last = supported[-1]
    out = list(theta)
    for b in range(last + 1, len(out)):
        out[b] = theta[last] + prior[b] - prior[last]
    return _project(out, fixed)


def predict(o: Observation, curve: list[float]) -> float:
    return o.tiv * sum(f * curve[b] for b, f in o.bins)


def residuals(obs: list[Observation], curve: list[float]) -> tuple[list[float], dict[str, float]]:
    """County log errors (both sides positive, material rows only) and
    event-total log errors (every row)."""
    county: list[float] = []
    ev_m: dict[str, float] = {}
    ev_a: dict[str, float] = {}
    for o in obs:
        m = predict(o, curve)
        if m > 0 and o.actual > 0 and o.upper_bound is None:
            county.append(math.log(m / o.actual))
        ev_m[o.event] = ev_m.get(o.event, 0.0) + m
        ev_a[o.event] = ev_a.get(o.event, 0.0) + o.actual
    events = {
        e: math.log(ev_m[e] / ev_a[e]) for e in ev_m if ev_m[e] > 0 and ev_a[e] > 0
    }
    return county, events


def _rms(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def fit_with_holdout(
    obs: list[Observation],
    prior_curve: list[float],
    opts: FitOptions | None = None,
) -> FitResult:
    """Fit on everything, then refit leaving each event out to score it unseen."""
    full = fit_curve(obs, prior_curve, opts)
    events = full.events
    if len(events) < 2:
        return full
    oos_county: list[float] = []
    for held in events:
        train = [o for o in obs if o.event != held]
        test = [o for o in obs if o.event == held]
        fitted = fit_curve(train, prior_curve, opts)
        full.oos_curves[held] = fitted.curve
        county, ev = residuals(test, fitted.curve)
        oos_county.extend(county)
        if held in ev:
            full.oos_event_errors[held] = ev[held]
    full.oos_county_sigma = _rms(oos_county)
    return full
