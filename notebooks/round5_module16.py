"""
Round 5 queries. Three readings now bracket Function 1's peak, so a curve
through them replaces stepping and hoping.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

SEED, N_CAND = 0, 50_000
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
LOG_Y, NOISY = {5}, {2}


def load(i):
    return (np.atleast_2d(np.load(f"data/function_{i}/inputs.npy")),
            np.ravel(np.load(f"data/function_{i}/outputs.npy")))


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
    n = len(y); ge, me = [], []
    for j in range(n):
        m = np.ones(n, bool); m[j] = False
        g = make_gp(i, X.shape[1]).fit(X[m], fwd(i, y[m]))
        ge.append((g.predict(X[j:j+1])[0] - fwd(i, y[j])) ** 2)
        me.append((fwd(i, y[m]).mean() - fwd(i, y[j])) ** 2)
    return np.sqrt(np.mean(ge)), np.sqrt(np.mean(me))


print("=" * 78)
print("FUNCTION 1: the old target turned out to be downhill")
print("=" * 78)
X1, y1 = load(1)
L = np.log10(np.abs(y1))
P0 = np.array([0.577271, 0.584651])          # two rounds ago
P1 = np.array([0.650114, 0.681526])          # the old target, now known to be weak
best_i = L.argmax()
print(f"strongest reading is now {np.round(X1[best_i],6)} at log10|y| = {L[best_i]:.2f}, "
      f"sign {'+' if y1[best_i] > 0 else '-'}")
print(f"the old target {np.round(P1,6)} sits at {L[np.argmin(np.linalg.norm(X1-P1,axis=1))]:.2f}, "
      "so that direction now goes down\n")

# Candidate A: three readings lie on one line, so fit a parabola along it
ts, ls = [], []
for j in range(len(X1)):
    v = X1[j] - P0; d = P1 - P0
    t = np.dot(v, d) / np.dot(d, d)
    if np.linalg.norm(v - t * d) < 0.02:
        ts.append(t); ls.append(L[j])
ts, ls = np.array(ts), np.array(ls)
# keep only the readings that bracket the peak. The one at t = -0.5 is eleven
# decades down and belongs to a different regime, and including it drags the fit.
keep = ts >= -0.05
ts, ls = ts[keep], ls[keep]
c = np.polyfit(ts, ls, 2)
t_star = -c[1] / (2 * c[0])
cand_A = P0 + t_star * (P1 - P0)
predA = np.polyval(c, t_star)
print(f"A  parabola along that line through {len(ts)} readings")
print(f"   peak at t={t_star:.3f} -> {np.round(cand_A,6)}, expects log10|y| ~ {predA:+.2f}")
print(f"   that is only {np.linalg.norm(cand_A - X1[best_i]):.3f} from the best point, "
      f"so about {10**(predA - L[best_i]):.1f}x upside")

# Candidate B: refit the 2D surface on all readings and take its peak
x1, x2 = X1[:, 0], X1[:, 1]
A = np.column_stack([np.ones_like(x1), x1, x2, x1**2, x2**2, x1*x2])
q, *_ = np.linalg.lstsq(A, L, rcond=None)
r2 = 1 - ((L - A @ q)**2).sum() / ((L - L.mean())**2).sum()
f = lambda p: -(q[0] + q[1]*p[0] + q[2]*p[1] + q[3]*p[0]**2 + q[4]*p[1]**2 + q[5]*p[0]*p[1])
bb = (np.inf, None)
for s in np.random.default_rng(SEED).random((80, 2)):
    r = minimize(f, s, bounds=[(0, 1)] * 2)
    if r.fun < bb[0]:
        bb = (r.fun, r.x)
cand_B, predB = np.clip(bb[1], 0, 1), -bb[0]
print(f"B  2D surface refit on all {len(X1)} readings, R2={r2:.3f}")
print(f"   peak at {np.round(cand_B,6)}, expects log10|y| ~ {predB:+.2f}, "
      f"{np.linalg.norm(cand_B - X1[best_i]):.3f} from the best point")

print("\nMy last two predictions on this function under-shot by about three decades each,")
print("so I do not trust either number, only the locations. A sits between two positive")
print("readings and is a short step; B extrapolates past everything I have measured.")
print("Taking A.")
queries = {1: cand_A}

print()
print("=" * 78)
print("FUNCTIONS 2 TO 8")
print("=" * 78)
rng = np.random.default_rng(SEED)
for i in range(2, 9):
    d = DIMS[i]
    X, y = load(i)
    yt = fwd(i, y)
    g_rmse, m_rmse = loo_trust(i, X, y)
    trust = g_rmse < m_rmse
    gp = make_gp(i, d).fit(X, yt)
    cand = rng.random((N_CAND, d))
    mu, sd = gp.predict(cand, return_std=True)
    sd = np.maximum(sd, 1e-12)
    if trust:
        z = (mu - yt.max()) / sd
        acq = (mu - yt.max()) * norm.cdf(z) + sd * norm.pdf(z)
    else:
        acq = mu + 2.0 * sd
    j = int(np.argmax(acq))
    queries[i] = cand[j]
    ls_ = np.atleast_1d(gp.kernel_.k1.k2.length_scale if i in NOISY
                        else gp.kernel_.k2.length_scale)
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {g_rmse:7.3f} vs mean {m_rmse:7.3f}  "
          f"{'trust' if trust else 'DISTRUST'}  "
          f"{'exploration' if sd[j] > np.median(sd) else 'exploitation'}  "
          f"ls={np.round(ls_,2)}")

print()
print("=" * 78)
lines = [f"Function {i}: " + "-".join(f"{v:.6f}" for v in queries[i]) for i in range(1, 9)]
print("\n".join(lines))
open("submissions/round5_module16.txt", "w").write("\n".join(lines) + "\n")
