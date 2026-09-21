"""
Round 9 queries.

Two changes from round 8:

Function 1 has stopped moving. The round 8 query landed between two known points
and came back at 1.98, the best reading of the run, and both axes now peak at
points already measured: across the ridge at -0.0144, along it at +0.0003. The
parabola cannot suggest anything new because the sample spacing is now about the
width of the peak itself. So this round halves the bracket instead of guessing a
peak, which either finds a better point or confirms the current one is the top.

Function 4 gets a trust region. It passed the leave-one-out check again and then
returned -8.85, its worst reading yet. That is the third time the global model has
sent it somewhere far from anything measured and been badly wrong. Restricting its
candidates to a box around its best point is the fix the TuRBO paper describes.
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

SEED, N_CAND = 0, 50_000
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
LOG_Y, NOISY = {5}, {2}
TRUST_REGION = {4: 0.15}      # half-width of the box, per input, around the best point


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
print("FUNCTION 1: converged to the sample spacing, so halve the bracket")
print("=" * 78)
anchor = np.array([0.617938, 0.638734])       # the round 7 best, origin of the offsets
P0 = np.array([0.577271, 0.584651]); P1 = np.array([0.650114, 0.681526])
u = (P1 - P0) / np.linalg.norm(P1 - P0)
perp = np.array([-u[1], u[0]])

# Across the ridge. The middle reading is the highest, so the peak is between the
# outer two, but a parabola through these three just points back at the middle one.
across = [(-0.0300, 1.1142113638492601),
          (-0.0144, 1.9846898128688588),
          ( 0.0000, 1.1838076163353304)]
sa = np.array([a[0] for a in across]); La = np.log10([abs(a[1]) for a in across])
ca = np.polyfit(sa, La, 2); peak_across = -ca[1] / (2 * ca[0])

# Along the ridge, all taken at across = 0. Note the sign flip at +0.0535: the ridge
# ends just past the current best, which is why stepping further along it is a bad idea.
along = [(-0.0192, 0.542672),
         ( 0.0000, 1.1838076163353304),
         ( 0.0535, -0.00360606)]
sl = np.array([a[0] for a in along]); Ll = np.log10([abs(a[1]) for a in along])
cl = np.polyfit(sl, Ll, 2); peak_along = -cl[1] / (2 * cl[0])

print(f"across: peak at {peak_across:+.4f}, and -0.0144 is already sampled")
print(f"along:  peak at {peak_along:+.4f}, and  0.0000 is already sampled")
print("Both estimates sit on points I hold, so neither suggests a new query.")

# Halve the wider untested gap. The parabola tips very slightly left of -0.0144,
# so take the midpoint between -0.0300 and -0.0144.
new_across = (-0.0300 + -0.0144) / 2
cand = anchor + peak_along * u + new_across * perp
assert -0.0300 < new_across < 0.0000, "query left the measured bracket"
print(f"querying across {new_across:+.4f}, along {peak_along:+.4f}  ->  {np.round(cand, 6)}")
print("This sits between two readings I hold. It is a bracket check, not a peak guess.")
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

    if i in TRUST_REGION:
        # Draw candidates from a box around the best point instead of the whole space.
        best_x = X[int(np.argmax(yt))]
        half = TRUST_REGION[i]
        lo = np.clip(best_x - half, 0.0, 0.999999)
        hi = np.clip(best_x + half, 0.0, 0.999999)
        c_pts = rng.uniform(lo, hi, size=(N_CAND, dd))
    else:
        c_pts = rng.random((N_CAND, dd))

    mu, sd = gp.predict(c_pts, return_std=True)
    sd = np.maximum(sd, 1e-12)
    if trust:
        z = (mu - yt.max()) / sd
        acq = (mu - yt.max()) * norm.cdf(z) + sd * norm.pdf(z)
    else:
        acq = mu + 2.0 * sd
    j = int(np.argmax(acq))
    queries[i] = c_pts[j]
    box = "  [trust region]" if i in TRUST_REGION else ""
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {g_rmse:7.3f} vs mean {m_rmse:7.3f}  "
          f"{'trust' if trust else 'DISTRUST'}  "
          f"{'exploration' if sd[j] > np.median(sd) else 'exploitation'}{box}")

print()
lines = [f"Function {i}: " + "-".join(f"{v:.6f}" for v in queries[i]) for i in range(1, 9)]
print("\n".join(lines))
open("submissions/round9_module20.txt", "w").write("\n".join(lines) + "\n")
