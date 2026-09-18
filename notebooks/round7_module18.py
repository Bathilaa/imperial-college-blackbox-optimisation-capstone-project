"""
Module 18 / Round 7 query generation.

Round 6 probed 0.03 perpendicular to the Function 1 line and lost only 0.026
decades, against 2.5 decades for 0.054 along it. That reads as a ridge running
perpendicular to the direction walked for four rounds, so this round steps
0.12 along it.

NOTE, WRITTEN AFTER THE RESULT: this was wrong. The 0.03 probe was taken on one
side of the peak and the 0.12 step was taken on the other. The surface is not
symmetric; the far side falls off a cliff and this query returned 2.2e-13,
about thirteen decades below the best. See round8_module19.py.
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
print("FUNCTION 1: step along what looks like a ridge")
print("=" * 78)
X1, y1 = load(1)
L = np.log10(np.abs(y1))
P0 = np.array([0.577271, 0.584651]); P1 = np.array([0.650114, 0.681526])
u = (P1 - P0) / np.linalg.norm(P1 - P0)
perp = np.array([-u[1], u[0]])
best = np.array([0.617938, 0.638734])
STEP = 0.12
cand = np.clip(best + STEP * perp, 0, 1)
print(f"best reading {np.round(best,6)}, log10|y| = +0.073")
print("Round 6 went 0.030 the other way along this axis and lost 0.026 decades.")
print(f"Stepping +{STEP} here, on the assumption the axis is roughly symmetric.")
print("That assumption is the weak point: only one side has been measured.")
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
open("submissions/round7_module18.txt", "w").write("\n".join(lines) + "\n")
