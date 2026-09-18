"""
Round 8 queries. Three readings bracket Function 1's sideways axis now, so
this interpolates between them instead of stepping past them.
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
print("FUNCTION 1: interpolate between readings, stop extrapolating past them")
print("=" * 78)
best = np.array([0.617938, 0.638734])
P0 = np.array([0.577271, 0.584651]); P1 = np.array([0.650114, 0.681526])
u = (P1 - P0) / np.linalg.norm(P1 - P0)
perp = np.array([-u[1], u[0]])
# the three readings taken across this axis, as offsets from the best point
obs = [(-0.03, 1.1142113638492601), (0.0, 1.1838076163353304),
       (0.12, 2.2128269235845376e-13)]
s = np.array([o[0] for o in obs]); L = np.log10([abs(o[1]) for o in obs])
c = np.polyfit(s, L, 2)
st = -c[1] / (2 * c[0])
cand = best + st * perp
print(f"offsets {list(s)} gave log10|y| {list(np.round(L,3))}")
print(f"curve peaks at offset {st:+.4f} -> {np.round(cand,6)}")
print(f"expects log10|y| {np.polyval(c,st):+.3f} against the best {L[1]:+.3f}")
print("This sits between two readings already held, so it interpolates.")
print("Last week's failure was an extrapolation past everything measured.")
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
open("submissions/round8_module19.txt", "w").write("\n".join(lines) + "\n")
