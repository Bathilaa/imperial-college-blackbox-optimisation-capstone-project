"""
Round 10 queries.

Function 1: the round 9 bracket check worked. Querying across -0.0222 returned 1.72,
below the 1.98 at -0.0144, which confirms the peak sits between -0.0222 and 0.

Refitting on the three tight readings puts the across peak at -0.01456, which is
0.00016 from a reading I already hold. The along peak is +0.00032, which is 0.00032
from one I already hold. Both axes have converged, so no curve fit can suggest an
improvement any more.

That leaves one question worth a query. Both scans were single lines through different
centres, so I know the best point on each line but not whether the ridge is tilted. This
round steps along the ridge while holding the converged across offset. The along curve
says a step of +0.0103 should return about 1.62 if the ridge is straight. If it comes
back near 1.98 or above, the ridge is tilted and there is more to find. Either answer is
worth knowing and the predicted cost is modest.

Function 4: the trust region helped and is kept, but tightened. Round 9 returned -1.26
against -8.85 the round before, so restricting the search cut the damage by a factor of
seven. It is still short of the 0.63 best, so the box shrinks from 0.15 to 0.10, which
is what TuRBO does after a round that fails to improve.
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

SEED, N_CAND = 0, 50_000
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
LOG_Y, NOISY = {5}, {2}
TRUST_REGION = {4: 0.10}          # shrunk from 0.15 after round 9 failed to improve


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
print("FUNCTION 1: both axes converged, so test whether the ridge is tilted")
print("=" * 78)
anchor = np.array([0.617938, 0.638734])
P0 = np.array([0.577271, 0.584651]); P1 = np.array([0.650114, 0.681526])
u = (P1 - P0) / np.linalg.norm(P1 - P0)
perp = np.array([-u[1], u[0]])

# Across the ridge: the three tight readings that enclose the peak.
sa = np.array([-0.0222, -0.0144, 0.0])
La = np.log10([1.7218485807793023, 1.9846898128688588, 1.1838076163353304])
ca = np.polyfit(sa, La, 2); peak_across = -ca[1] / (2 * ca[0])

# Along the ridge: three readings, all taken at across = 0. Note the sign flip at
# +0.0535, so the ridge ends not far past the best point.
sl = np.array([-0.0192, 0.0, 0.0535])
Ll = np.log10([0.542672, 1.1838076163353304, 0.00360606])
cl = np.polyfit(sl, Ll, 2); peak_along = -cl[1] / (2 * cl[0])

print(f"across peak {peak_across:+.5f}, which is {abs(peak_across + 0.0144):.5f} from a reading held")
print(f"along  peak {peak_along:+.5f}, which is {abs(peak_along):.5f} from a reading held")
print("Both converged. No fit can propose an improvement, so this tests the shape instead.")

STEP = 0.0103                      # moderate, well inside the measured along range
new_along = peak_along + STEP
assert sl.min() < new_along < sl.max(), "step left the measured along range"
cand = anchor + new_along * u + peak_across * perp
ratio = 10 ** (np.polyval(cl, new_along) - np.polyval(cl, peak_along))
print(f"querying along {new_along:+.4f}, across {peak_across:+.5f}  ->  {np.round(cand, 6)}")
print(f"predicts about {1.9846898128688588 * ratio:.3f} if the ridge is straight")
print("Coming back near 1.98 or above would mean the ridge is tilted and worth following.")
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
    box = f"  [box {TRUST_REGION[i]}]" if i in TRUST_REGION else ""
    print(f"F{i}  {'EI ' if trust else 'UCB'}  LOO {g_rmse:7.3f} vs mean {m_rmse:7.3f}  "
          f"{'trust' if trust else 'DISTRUST'}  "
          f"{'exploration' if sd[j] > np.median(sd) else 'exploitation'}{box}")

print()
lines = [f"Function {i}: " + "-".join(f"{v:.6f}" for v in queries[i]) for i in range(1, 9)]
print("\n".join(lines))
open("submissions/round10_module21.txt", "w").write("\n".join(lines) + "\n")
