"""
Module 14 / Round 3 query generation.

Function 1 is handled separately again. Two candidates are worked out and
compared, and the reasoning for the choice is printed.
Functions 2 to 8 use the same GP plus leave-one-out trust check as Round 2.
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


# ---------------------------------------------------------------- Function 1
print("=" * 78)
print("FUNCTION 1: two candidates compared")
print("=" * 78)
X1, y1 = load(1)
L = np.log10(np.abs(y1))

# how last week's fit actually performed
x, z = X1[:-1, 0], X1[:-1, 1]
A = np.column_stack([np.ones_like(x), x, z, x**2, z**2, x*z])
c_old, *_ = np.linalg.lstsq(A, L[:-1], rcond=None)
q = X1[-1]
pred_old = (c_old[0] + c_old[1]*q[0] + c_old[2]*q[1] + c_old[3]*q[0]**2
            + c_old[4]*q[1]**2 + c_old[5]*q[0]*q[1])
print(f"last week the curve predicted log10|y| = {pred_old:+.1f} at {np.round(q,6)}")
print(f"the reading that came back was          {L[-1]:+.1f}  (y = {y1[-1]:+.4g})")
print("so the curve found a better spot but its size prediction was far out.\n")

# candidate A: refit the curve on all 12 points and take its new peak
x, z = X1[:, 0], X1[:, 1]
A = np.column_stack([np.ones_like(x), x, z, x**2, z**2, x*z])
c_new, *_ = np.linalg.lstsq(A, L, rcond=None)
r2 = 1 - ((L - A @ c_new)**2).sum() / ((L - L.mean())**2).sum()
f = lambda p: -(c_new[0] + c_new[1]*p[0] + c_new[2]*p[1] + c_new[3]*p[0]**2
                + c_new[4]*p[1]**2 + c_new[5]*p[0]*p[1])
best = (np.inf, None)
for s in np.random.default_rng(SEED).random((80, 2)):
    r = minimize(f, s, bounds=[(0, 1)] * 2)
    if r.fun < best[0]:
        best = (r.fun, r.x)
cand_A, predA = np.clip(best[1], 0, 1), -best[0]

# candidate B: step from the best positive reading toward the strongest reading
pos = y1 > 0
p_best = X1[np.where(pos)[0][y1[pos].argmax()]]     # biggest positive value
m_best = X1[L.argmax()]                              # biggest size, but negative
STEP = 1 / 3
cand_B = p_best + STEP * (m_best - p_best)
predB = L[np.where(pos)[0][y1[pos].argmax()]] + STEP * (L.max() - L[np.where(pos)[0][y1[pos].argmax()]])

print(f"A  refit the curve, query its new peak : {np.round(cand_A,6)}  "
      f"fit R2={r2:.3f}, expects log10|y| ~ {predA:+.1f}")
print(f"   it sits {np.linalg.norm(cand_A - q):.3f} from last week's query, so a small step")
print(f"B  step 1/3 of the way from the best positive reading {np.round(p_best,6)}")
print(f"   toward the strongest reading {np.round(m_best,6)} (which is negative)")
print(f"   gives {np.round(cand_B,6)}, {np.linalg.norm(cand_B - p_best):.3f} away, "
      f"expects log10|y| ~ {predB:+.1f} if the sign holds")
print()
print("nearest known sign flip is 0.096 apart, so a 0.06 step should stay positive.")
print("A gains about one order of magnitude, B about three. Going with B.")
queries = {1: cand_B}

# ------------------------------------------------------------ Functions 2-8
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
        zz = (mu - yt.max()) / sd
        acq = (mu - yt.max()) * norm.cdf(zz) + sd * norm.pdf(zz)
    else:
        acq = mu + 2.0 * sd
    j = int(np.argmax(acq))
    queries[i] = cand[j]
    ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale if i in NOISY
                       else gp.kernel_.k2.length_scale)
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {g_rmse:7.3f} vs mean {m_rmse:7.3f}  "
          f"{'trust' if trust else 'DISTRUST'}  "
          f"{'exploration' if sd[j] > np.median(sd) else 'exploitation'}  "
          f"ls={np.round(ls,2)}")

print()
print("=" * 78)
lines = [f"Function {i}: " + "-".join(f"{v:.6f}" for v in queries[i]) for i in range(1, 9)]
print("\n".join(lines))
open("submissions/round3_module14.txt", "w").write("\n".join(lines) + "\n")
print("\nsaved submissions/round3_module14.txt")
