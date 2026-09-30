"""
Rounds 11, 12 and 13 queries. Planned but not submitted: I ran out of time, so round 10
was my last submitted round. The planned queries are written to planned/, not submissions/.

All three were due on the same day, so rounds 12 and 13 could not wait for results. Each
round after 11 treats the model's prediction for the previous round as if it were the
real result and refits. That is the "kriging believer" trick for batch Bayesian
optimisation: it stops the three rounds landing on the same point, because the model
becomes certain about any point it has already "seen".

Round 10 improved nothing. That is the second round in four with no gain.

Function 1: round 10 stepped along the ridge and got 1.38, below the 1.62 the along curve
predicted, so the ridge is not tilted in that direction. Instead of two separate line fits,
round 11 fits one quadratic surface (in log size) to every reading near the peak and
queries its top, still only between readings already held. Rounds 12 and 13 bracket that
point along the direction the surface is least sure about.

Function 4: round 10 returned -2.27 inside the 0.10 box, worse than -1.26 the round before.
The box shrinks again, to 0.07, following the same rule as last time.
"""
import time
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

t0 = time.time()
SEED, N_CAND = 0, 50_000
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
LOG_Y, NOISY = {5}, {2}
TRUST_REGION = {4: 0.07}          # shrunk from 0.10 after round 10 failed to improve
ROUNDS = (11, 12, 13)


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


queries = {r: {} for r in ROUNDS}

# ---------------------------------------------------------------- Function 1
print("=" * 78)
print("FUNCTION 1: one quadratic surface through the readings near the peak")
print("=" * 78)
X1, y1 = load(1)
best = X1[int(np.argmax(y1))]
P0 = np.array([0.577271, 0.584651]); P1 = np.array([0.650114, 0.681526])
u = (P1 - P0) / np.linalg.norm(P1 - P0)            # along the ridge
perp = np.array([-u[1], u[0]])                       # across the ridge

near = np.linalg.norm(X1 - best, axis=1) < 0.04
Xn, Ln = X1[near], np.log10(np.abs(y1[near]))
s = (Xn - best) @ u; t = (Xn - best) @ perp


def feats(s, t):
    return np.column_stack([np.ones_like(s), s, t, s * s, t * t, s * t])


coef, *_ = np.linalg.lstsq(feats(s, t), Ln, rcond=None)
a, b, c, d, e, f = coef
H = np.array([[2 * d, f], [f, 2 * e]])               # curvature of the surface
evals, evecs = np.linalg.eigh(H)
print(f"{near.sum()} readings used, curvature eigenvalues {np.round(evals, 1)}")
assert np.all(evals < 0), "surface is not a hill, do not trust its top"

top = np.linalg.solve(H, -np.array([b, c]))         # where the slope is zero
lo_s, hi_s, lo_t, hi_t = s.min(), s.max(), t.min(), t.max()
top = np.array([np.clip(top[0], lo_s, hi_s), np.clip(top[1], lo_t, hi_t)])
flat = evecs[:, np.argmax(evals)]                    # least curved direction
DELTA = 0.004
pts = [top, top + DELTA * flat, top - DELTA * flat]
for r, (ps, pt) in zip(ROUNDS, pts):
    ps = np.clip(ps, lo_s, hi_s); pt = np.clip(pt, lo_t, hi_t)
    x = best + ps * u + pt * perp
    pred = 10 ** (feats(np.array([ps]), np.array([pt])) @ coef)[0]
    queries[r][1] = x
    print(f"round {r}: along {ps:+.5f} across {pt:+.5f} -> {np.round(x, 6)}  predicts {pred:.3f}")
print(f"best held so far {y1.max():.3f}")

# ------------------------------------------------------------ Functions 2 to 8
print()
print("=" * 78)
print("FUNCTIONS 2 TO 8, three rounds each, later rounds believe the model's prediction")
print("=" * 78)
rng = np.random.default_rng(SEED)
for i in range(2, 9):
    dd = DIMS[i]
    X, y = load(i)
    yt = fwd(i, y)
    g_rmse, m_rmse = loo_trust(i, X, y)
    trust = g_rmse < m_rmse
    gp = make_gp(i, dd).fit(X, yt)
    kern = gp.kernel_
    Xf, yf = X.copy(), yt.copy()
    for r in ROUNDS:
        if r > ROUNDS[0]:
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                          optimizer=None).fit(Xf, yf)
        if i in TRUST_REGION:
            bx = X[int(np.argmax(yt))]
            h = TRUST_REGION[i]
            c_pts = rng.uniform(np.clip(bx - h, 0, 0.999999), np.clip(bx + h, 0, 0.999999),
                                size=(N_CAND, dd))
        else:
            c_pts = rng.random((N_CAND, dd))
        mu, sd = gp.predict(c_pts, return_std=True)
        sd = np.maximum(sd, 1e-12)
        if trust:
            z = (mu - yf.max()) / sd
            acq = (mu - yf.max()) * norm.cdf(z) + sd * norm.pdf(z)
        else:
            acq = mu + 2.0 * sd
        # Function 2 is noisy, so believing a prediction never makes the model fully sure
        # about that point. Rule out anything within 0.05 of this batch's earlier rounds.
        for q in ROUNDS:
            if q < r:
                acq[np.linalg.norm(c_pts - queries[q][i], axis=1) < 0.05] = -np.inf
        j = int(np.argmax(acq))
        x = c_pts[j]
        queries[r][i] = x
        Xf = np.vstack([Xf, x]); yf = np.append(yf, mu[j])     # believe the prediction
    gaps = [np.linalg.norm(queries[a][i] - queries[b][i]) for a, b in ((11, 12), (12, 13), (11, 13))]
    box = f"  [box {TRUST_REGION[i]}]" if i in TRUST_REGION else ""
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {g_rmse:7.3f} vs mean {m_rmse:7.3f}  "
          f"{'trust' if trust else 'DISTRUST'}  min gap between rounds {min(gaps):.3f}{box}")

print()


def fmt(v):
    return f"{min(max(v, 0.0), 0.999999):.6f}"      # portal rejects anything that rounds to 1


for r, mod in zip(ROUNDS, (22, 23, 24)):
    lines = [f"Function {i}: " + "-".join(fmt(v) for v in queries[r][i]) for i in range(1, 9)]
    open(f"planned/round{r}_module{mod}.txt", "w").write("\n".join(lines) + "\n")
    print(f"--- round {r} (module {mod}) ---")
    print("\n".join(lines))
print(f"\ntook {time.time() - t0:.0f}s")
