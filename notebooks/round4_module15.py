"""
Round 4 queries. Function 1 has little upside left on its line, so the step shrinks.
"""
import numpy as np
from scipy.stats import norm
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
print("FUNCTION 1: the upside has collapsed, so the step shrinks")
print("=" * 78)
X1, y1 = load(1)
L = np.log10(np.abs(y1))
pos = y1 > 0
i_pos = np.where(pos)[0][y1[pos].argmax()]
p_best, m_best = X1[i_pos], X1[L.argmax()]

print(f"Round 3 step gained {L[i_pos] - (-11.06):.1f} decades and kept a positive sign.")
print(f"But the strongest reading anywhere is only {L.max() - L[i_pos]:.2f} decades above")
print(f"where I now sit, so at most about {10 ** (L.max() - L[i_pos]):.0f}x remains on this line,")
print("against the 13 million x just gained. And that reading is negative, so the")
print("sign must flip somewhere in the gap.\n")

for t in (0.4, 1.0):
    cand = p_best + t * (m_best - p_best)
    pred = L[i_pos] + t * (L.max() - L[i_pos])
    print(f"  t={t:.1f}  {np.round(cand,6)}  step {np.linalg.norm(cand-p_best):.3f}  "
          f"expects log10|y| ~ {pred:.2f}")
print("\nSign flips seen elsewhere are 0.05 to 0.10 apart. A 0.048 step (t=0.4) stays")
print("inside one lobe on that evidence; a full step almost certainly crosses it.")
print("Small remaining upside plus real sign risk means the cautious step wins.")
queries = {1: p_best + 0.4 * (m_best - p_best)}

print()
print("=" * 78)
print("FUNCTIONS 2 TO 8")
print("=" * 78)
rng = np.random.default_rng(SEED)
diag = {}
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
    ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale if i in NOISY
                       else gp.kernel_.k2.length_scale)
    diag[i] = (trust, ls)
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {g_rmse:7.3f} vs mean {m_rmse:7.3f}  "
          f"{'trust' if trust else 'DISTRUST'}  "
          f"{'exploration' if sd[j] > np.median(sd) else 'exploitation'}  "
          f"ls={np.round(ls,2)}")

print("\ninputs whose length scale is pinned at the ceiling, so the model says they do nothing:")
for i, (t, ls) in diag.items():
    dead = [k + 1 for k, v in enumerate(ls) if v >= 99]
    if dead:
        print(f"  F{i}: x{dead}")

print()
print("=" * 78)
lines = [f"Function {i}: " + "-".join(f"{v:.6f}" for v in queries[i]) for i in range(1, 9)]
print("\n".join(lines))
open("submissions/round4_module15.txt", "w").write("\n".join(lines) + "\n")
print("\nsaved submissions/round4_module15.txt")
