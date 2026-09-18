"""
Round 6 queries. Function 1's line is done, so this probes sideways.
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
print("FUNCTION 1: the line is finished, so probe sideways")
print("=" * 78)
X1, y1 = load(1)
L = np.log10(np.abs(y1))
P0 = np.array([0.577271, 0.584651]); P1 = np.array([0.650114, 0.681526])
d = P1 - P0; u = d / np.linalg.norm(d); perp = np.array([-u[1], u[0]])
best = X1[L.argmax()]

# the 2D surface fit is the only thing carrying any sideways information
A = np.column_stack([np.ones(len(X1)), X1[:, 0], X1[:, 1],
                     X1[:, 0]**2, X1[:, 1]**2, X1[:, 0]*X1[:, 1]])
q, *_ = np.linalg.lstsq(A, L, rcond=None)
f = lambda p: -(q[0] + q[1]*p[0] + q[2]*p[1] + q[3]*p[0]**2 + q[4]*p[1]**2 + q[5]*p[0]*p[1])
bb = (np.inf, None)
for s in np.random.default_rng(SEED).random((80, 2)):
    r = minimize(f, s, bounds=[(0, 1)] * 2)
    if r.fun < bb[0]:
        bb = (r.fun, r.x)
side = np.dot(np.clip(bb[1], 0, 1) - best, perp)
sign = 1.0 if side >= 0 else -1.0
print(f"the 2D surface fit puts its peak {abs(side):.3f} to the "
      f"{'+' if sign > 0 else '-'} side of the line, so probe that way")

STEP = 0.03
cand = np.clip(best + sign * STEP * perp, 0, 1)
print(f"going 0.03 sideways from the best point: {np.round(cand,6)}")
print("along the line, 0.054 costs 2.5 decades, so 0.03 should cost under one")
print("decade if this really is the peak, and gain if the ridge runs across.")
queries = {1: cand}

print()
print("=" * 78)
print("FUNCTIONS 2 TO 8")
print("=" * 78)
rng = np.random.default_rng(SEED)
for i in range(2, 9):
    dd = DIMS[i]
    X, y = load(i)
    yt = fwd(i, y)
    g_rmse, m_rmse = loo_trust(i, X, y)
    trust = g_rmse < m_rmse
    gp = make_gp(i, dd).fit(X, yt)
    c = rng.random((N_CAND, dd))
    mu, sd = gp.predict(c, return_std=True)
    sd = np.maximum(sd, 1e-12)
    if trust:
        z = (mu - yt.max()) / sd
        acq = (mu - yt.max()) * norm.cdf(z) + sd * norm.pdf(z)
    else:
        acq = mu + 2.0 * sd
    j = int(np.argmax(acq))
    queries[i] = c[j]
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {g_rmse:7.3f} vs mean {m_rmse:7.3f}  "
          f"{'trust' if trust else 'DISTRUST'}  "
          f"{'exploration' if sd[j] > np.median(sd) else 'exploitation'}")

print()
lines = [f"Function {i}: " + "-".join(f"{v:.6f}" for v in queries[i]) for i in range(1, 9)]
print("\n".join(lines))
open("submissions/round6_module17.txt", "w").write("\n".join(lines) + "\n")
