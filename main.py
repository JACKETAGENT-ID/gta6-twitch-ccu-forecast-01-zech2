"""gta6-twitch-ccu-forecast-01 entry (zech, v2): launch-day shape regression.

Upgrades over the v1 entry (per-hour log-linear regression on log p1 +
log channels_1, practice loss 0.184, rank #2508):

  * Hour-of-day basis: every forecast hour h = 2..24 gets its own fit, so
    each hour carries its own intercept - the launch-day curve's ramp
    (US unlocks at 05:00/08:00 UTC), evening peak (17:00-21:00 UTC) and
    late decay are learned per hour, not imposed.
  * Launch-day shape features: the hour just before the UTC release day
    (23:00, when the rolling-midnight unlock has already lit up New
    Zealand / Europe / Asia) against the hour-1 print, as two features:
    the signed dip log(v_{-1} / p1) and the evening overhang
    max(0, log(v_{-1} / p1)). Games whose hour 1 dipped hard against the
    prior evening rebound differently through the day; the pair carries
    that signal while staying flat when hour 1 already passed the evening
    (a simultaneous-global unlock's shape, not a dip).
  * Robust fitting: each hour's fit is ridge regression (lambda 0.5, which
    stabilises the collinear log-scale predictors) refined by Huber IRLS
    (3 rounds), so one shape-outlier launch (a simultaneous-global
    unlock, a live-service spike) cannot drag an hour's coefficients.
  * Local-in-scale kernel: comparables are weighted by a Gaussian kernel
    (bandwidth 2 in log units) on their own hour-1 print vs this case's,
    cut off beyond one dex - games of a similar size share a launch-day
    shape; tiny launches (whose later hours explode as regions unlock)
    would otherwise drag the fit of a huge launch around.
  * Quantile calibration from the fit's own residuals: sigma_h is the
    kernel-weighted residual standard deviation of that hour's fit
    (degrees-of-freedom corrected, floored at 0.15 so the band is never
    overconfident, capped at 1.2); the five quantiles are
    median * exp(z_tau * sigma_h).

No coefficient is hardcoded: every fit is computed at runtime from the
comparables shipped inside the case file, so the same code adapts to any
case. Local leave-one-out validation on the kit's 37 full comparables
(each held out in turn, predicted from the rest, scored by the contest's
own pinball loss): mean loss 0.423, vs 0.526 for the v1 method on the
same protocol; on games with hour-1 print >= 100k (the scale of the
practice launches and of GTA VI) 0.206 vs 0.210, and on games >= 10k
0.252 vs 0.263. (An early experiment that clamped the dip at zero
scored 0.393 overall but slightly worse on the big-game subsets, so the
shipped model keeps both the signed dip and the floored overhang.)

Python 3.13, standard library only. No network, no randomness, no clock:
two runs on the same input write byte-identical output. Runtime is far
under the 8 second limit (23 tiny ridge fits over ~40 comparables).
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RIDGE_LAMBDA = 0.5
HUBER_ROUNDS = 3
KERNEL_BW = 2.0
KERNEL_CUTOFF = math.log(10)
SIGMA_FLOOR = 0.15
SIGMA_CAP = 1.2
MIN_ROWS = 5


def _parse_ts(text):
    """ISO 8601 timestamp (with Z) -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None when the matrix is (near-)singular.
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col] / piv
                for k in range(col, n + 1):
                    aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def ridge_wls(rows, weights, lam=RIDGE_LAMBDA):
    """Ridge-weighted least squares with intercept (no penalty on it).

    rows: list of (xs, y); weights: same length, non-negative. Returns
    (beta, residuals) or None when the system is singular.
    """
    p = len(rows[0][0]) + 1
    normal = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for (xs, y), w in zip(rows, weights):
        if w == 0.0:
            continue
        vec = [1.0] + list(xs)
        for i in range(p):
            rhs[i] += w * vec[i] * y
            for j in range(p):
                normal[i][j] += w * vec[i] * vec[j]
    for i in range(1, p):
        normal[i][i] += lam
    beta = solve_linear(normal, rhs)
    if beta is None:
        return None
    residuals = []
    for xs, y in rows:
        vec = [1.0] + list(xs)
        residuals.append(y - sum(beta[i] * vec[i] for i in range(p)))
    return beta, residuals


def huber_ridge(rows, weights, lam=RIDGE_LAMBDA, rounds=HUBER_ROUNDS):
    """Ridge fit refined by Huber IRLS: downweight residual outliers."""
    fit = ridge_wls(rows, weights, lam)
    if fit is None:
        return None
    beta, residuals = fit
    for _ in range(rounds):
        ordered = sorted(abs(r) for r in residuals)
        mad = ordered[len(ordered) // 2] or 0.1
        scale = 1.4826 * mad
        if scale <= 0:
            break
        tuned = [w * (1.0 if abs(r) <= 1.345 * scale else 1.345 * scale / abs(r))
                 for w, r in zip(weights, residuals)]
        nxt = ridge_wls(rows, tuned, lam)
        if nxt is None:
            break
        beta, residuals = nxt
    return beta, residuals


def extract_comparables(case):
    """Release-day curves of the case's comparables, aligned to hour 1.

    Each comparable contributes its hour-1 print (p1), its hour-1 channel
    count (c1, None when missing), its launch-day shape pair (dip,
    overhang) where dip = log(v_{-1} / p1) with v_{-1} the hour just
    before its release day and overhang = max(0, dip) (both None when
    that hour is missing), and its viewers for hours 1..24. A comparable
    without an hour-1 row contributes nothing and is skipped.
    """
    out = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        day_start = comp.get("day_start")
        hourly = comp.get("hourly") or []
        if not day_start or not hourly:
            continue
        try:
            t0 = _parse_ts(day_start)
        except ValueError:
            continue
        by_hour = {}
        for row in hourly:
            try:
                dt = _parse_ts(row["from"])
            except (KeyError, ValueError):
                continue
            k = int((dt - t0).total_seconds() // 3600)
            if -24 * 14 <= k < 24:
                v = row.get("viewers")
                if isinstance(v, (int, float)) and v > 0:
                    by_hour[k] = row
        if 0 not in by_hour:
            continue
        p1 = float(by_hour[0]["viewers"])
        c1_raw = by_hour[0].get("channels")
        c1 = float(c1_raw) if isinstance(c1_raw, (int, float)) and c1_raw > 0 else None
        dip = None
        overhang = None
        if -1 in by_hour:
            dip = math.log(max(1.0, float(by_hour[-1]["viewers"])) / max(1.0, p1))
            overhang = max(0.0, dip)
        series = {h: float(by_hour[h]["viewers"]) for h in range(0, 24) if h in by_hour}
        if len(series) < 12:
            continue
        out.append({"p1": p1, "c1": c1, "dip": dip, "overhang": overhang, "series": series})
    return out


def target_shape(case):
    """(dip, overhang) for the case itself, from its history_hourly.

    v_{-1} is the whole-hour mean of the hour just before the release day
    (day_start - 1h); falls back to the nearest available hour within 3h
    before day_start, else (None, None) (both shape features are then
    dropped).
    """
    try:
        t0 = _parse_ts(case["day_start"])
    except (KeyError, ValueError):
        return None, None
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        return None, None
    p1 = max(1.0, float(first["viewers"]))
    best = None
    for row in case.get("history_hourly") or []:
        try:
            dt = _parse_ts(row["from"])
        except (KeyError, ValueError):
            continue
        delta = (dt - t0).total_seconds() / 3600.0
        v = row.get("viewers")
        if not isinstance(v, (int, float)) or v <= 0:
            continue
        if -4 < delta <= -1:
            if best is None or abs(delta + 1) < abs(best[0] + 1):
                best = (delta, float(v))
    if best is None:
        return None, None
    dip = math.log(max(1.0, best[1]) / p1)
    return dip, max(0.0, dip)


def fit_hour(comps, hour, p1, c1, dip, overhang):
    """Median and log-spread for one forecast hour (hour is 0-indexed).

    Fits log(viewers_hour) on log(p1), log(channels_1), the dip and the
    evening overhang, across the comparables that have that hour, with a
    Gaussian kernel on log p1 and Huber-robust ridge regression, then
    evaluates the fit at this case's own hour-1 print, channel count and
    shape pair. Returns (median, sigma).
    """
    use_c1 = c1 is not None
    use_shape = dip is not None and overhang is not None
    rows = []
    for comp in comps:
        viewers = comp["series"].get(hour)
        if viewers is None:
            continue
        xs = [math.log(max(1.0, comp["p1"]))]
        if use_c1:
            if comp["c1"] is None:
                continue
            xs.append(math.log(max(1.0, comp["c1"])))
        if use_shape:
            if comp["dip"] is None or comp["overhang"] is None:
                continue
            xs.append(comp["dip"])
            xs.append(comp["overhang"])
        rows.append((tuple(xs), math.log(viewers)))
    if len(rows) < MIN_ROWS:
        return None
    log_p1 = math.log(max(1.0, p1))
    weights = []
    for (xs, _y) in rows:
        d = abs(xs[0] - log_p1)
        weights.append(math.exp(-((d) / KERNEL_BW) ** 2) if d <= KERNEL_CUTOFF else 0.0)
    if sum(1 for w in weights if w > 0.0) < MIN_ROWS:
        weights = [1.0] * len(rows)
    fitted = huber_ridge(rows, weights)
    if fitted is None:
        return None
    beta, residuals = fitted
    xs = [log_p1]
    if use_c1:
        xs.append(math.log(max(1.0, c1)))
    if use_shape:
        xs.append(dip)
        xs.append(overhang)
    mu = beta[0] + sum(beta[i + 1] * xs[i] for i in range(len(xs)))
    dof = max(1, len(residuals) - len(beta))
    sw = sum(weights)
    sigma = math.sqrt(
        sum(w * r * r for w, r in zip(weights, residuals))
        / max(1.0, sw * (dof / len(residuals)))
    )
    return math.exp(mu), min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def fallback_median_sigma(comps, hour, p1):
    """Median comparable ratio viewers_h / p1, spread from the log ratios."""
    ratios = [
        math.log(comp["series"][hour] / max(1.0, comp["p1"]))
        for comp in comps
        if hour in comp["series"]
    ]
    if not ratios:
        return max(1.0, p1), 0.5
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    mean_ratio = sum(ratios) / len(ratios)
    if len(ratios) > 1:
        sigma = math.sqrt(sum((r - mean_ratio) ** 2 for r in ratios) / (len(ratios) - 1))
    else:
        sigma = 0.5
    return math.exp(math.log(max(1.0, p1)) + median_ratio), min(
        SIGMA_CAP, max(SIGMA_FLOOR, sigma)
    )


def round2(x):
    """x rounded to 2 decimals, half up (same as the kit starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    c1_raw = first.get("channels")
    c1 = float(c1_raw) if isinstance(c1_raw, (int, float)) and c1_raw > 0 else None
    dip, overhang = target_shape(case)
    comps = extract_comparables(case)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        fitted = fit_hour(comps, h - 1, p1, c1, dip, overhang)
        if fitted is None:
            median, sigma = fallback_median_sigma(comps, h - 1, p1)
        else:
            median, sigma = fitted
        qs = []
        for z in Z:
            q = median * math.exp(z * sigma)
            q = min(1e9, max(0.01, q))
            qs.append(round2(q))
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
