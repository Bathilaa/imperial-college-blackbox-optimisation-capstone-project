"""
Round 2 queries. Added a leave-one-out check so a model has to beat
predicting the average before it gets to exploit. Function 1 drops the GP
for a curve fitted to the size of its readings.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

SEED = 0
N_CAND = 50_000
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
LOG_Y = {5}          # strictly positive, spans three orders of magnitude
NOISY = {2}          # the only function the brief calls noisy


def load(i):
    x = np.load(f"data/function_{i}/inputs.npy")
    y = np.load(f"data/function_{i}/outputs.npy")
    return np.atleast_2d(x), np.ravel(y)


def make_gp(i, d):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
    if i in NOISY:
        k = k + WhiteKernel(1e-2, (1e-6, 1e0))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=10, random_state=SEED)


def fwd(i, y):
    return np.log(y) if i in LOG_Y else y


def loo_trust(i, X, y):
    """Leave-one-out RMSE of the GP against a mean predictor."""
    n = len(y)
    gp_err, mean_err = [], []
    for j in range(n):
        m = np.ones(n, bool); m[j] = False
        g = make_gp(i, X.shape[1]).fit(X[m], fwd(i, y[m]))
        gp_err.append((g.predict(X[j:j + 1])[0] - fwd(i, y[j])) ** 2)
        mean_err.append((fwd(i, y[m]).mean() - fwd(i, y[j])) ** 2)
    return np.sqrt(np.mean(gp_err)), np.sqrt(np.mean(mean_err))


def acquire(gp, best, cand, use_ei, kappa=2.0):
    mu, sd = gp.predict(cand, return_std=True)
    sd = np.maximum(sd, 1e-12)
    if use_ei:
        z = (mu - best) / sd
        return (mu - best) * norm.cdf(z) + sd * norm.pdf(z), mu, sd
    return mu + kappa * sd, mu, sd


def function_1(X, y):
    """Fit log10|y| with a full quadratic and return its maximiser."""
    L = np.log10(np.abs(y))
    x1, x2 = X[:, 0], X[:, 1]
    A = np.column_stack([np.ones_like(x1), x1, x2, x1 ** 2, x2 ** 2, x1 * x2])
    c, *_ = np.linalg.lstsq(A, L, rcond=None)
    pred = A @ c
    r2 = 1 - ((L - pred) ** 2).sum() / ((L - L.mean()) ** 2).sum()

    def neg(p):
        a, b = p
        return -(c[0] + c[1]*a + c[2]*b + c[3]*a**2 + c[4]*b**2 + c[5]*a*b)

    best, bx = np.inf, None
    rng = np.random.default_rng(SEED)
    for s in rng.random((60, 2)):
        r = minimize(neg, s, bounds=[(0, 1)] * 2)
        if r.fun < best:
            best, bx = r.fun, r.x
    return np.clip(bx, 0, 1), r2, -best


rng = np.random.default_rng(SEED)
queries, notes = {}, {}

print("=" * 78)
print("HINDSIGHT: what the Round 1 model predicted vs what came back")
print("=" * 78)
for i in range(1, 9):
    X, y = load(i)
    if i == 1:
        print(f"F1  actual {y[-1]:+.3e}   (no GP used, see below)")
        continue
    g = make_gp(i, X.shape[1]).fit(X[:-1], fwd(i, y[:-1]))
    mu, sd = g.predict(X[-1:], return_std=True)
    act = fwd(i, y[-1])
    prev = fwd(i, y[:-1]).max()
    print(f"F{i}  predicted {mu[0]:+.4f} +/- {sd[0]:.4f}   actual {act:+.4f}   "
          f"prev best {prev:+.4f}   {'IMPROVED' if act > prev else 'no gain'}   "
          f"{'inside 1sd' if abs(act - mu[0]) <= sd[0] else 'OUTSIDE 1sd'}")

print()
print("=" * 78)
print("ROUND 2 QUERY SELECTION")
print("=" * 78)

# ---- Function 1: direct quadratic fit, no GP ----
X, y = load(1)
xq, r2, peak = function_1(X, y)
queries[1] = xq
notes[1] = f"quadratic fit to log10|y|, R2={r2:.3f}, predicted log10|y|={peak:.1f}"
print(f"F1  quadratic envelope fit  R2={r2:.3f}  peak log10|y|~{peak:.1f}  "
      f"x={np.round(xq,6)}")

# ---- Functions 2-8: GP with a leave-one-out trust check ----
for i in range(2, 9):
    d = DIMS[i]
    X, y = load(i)
    yt = fwd(i, y)
    gp_rmse, mean_rmse = loo_trust(i, X, y)
    trust = gp_rmse < mean_rmse
    gp = make_gp(i, d).fit(X, yt)
    cand = rng.random((N_CAND, d))
    acq, mu, sd = acquire(gp, yt.max(), cand, use_ei=trust)
    j = int(np.argmax(acq))
    queries[i] = cand[j]
    ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale if i in NOISY
                       else gp.kernel_.k2.length_scale)
    driver = "exploration" if sd[j] > np.median(sd) else "exploitation"
    notes[i] = (f"{'EI' if trust else 'UCB'}, LOO RMSE {gp_rmse:.3f} vs mean "
                f"{mean_rmse:.3f}, {driver}")
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {gp_rmse:7.3f} vs mean "
          f"{mean_rmse:7.3f}  {'trust' if trust else 'DISTRUST'}  {driver:12s}  "
          f"ls={np.round(ls,2)}")

print()
print("=" * 78)
print("PORTAL STRINGS")
print("=" * 78)
lines = []
for i in range(1, 9):
    s = "-".join(f"{v:.6f}" for v in queries[i])
    lines.append(f"Function {i}: {s}")
    print(f"Function {i}: {s}")

with open("submissions/round2_module13.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
np.save("submissions/round2_module13_queries.npy",
        np.array([np.pad(queries[i], (0, 8 - len(queries[i]))) for i in range(1, 9)]))
print("\nsaved submissions/round2_module13.txt")
