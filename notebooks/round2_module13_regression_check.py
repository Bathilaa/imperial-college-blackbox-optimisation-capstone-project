"""
Linear regression on each function, for the Module 13 write-up.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
print(f"{'F':>2} {'n':>3} {'d':>2} {'R2':>7} {'adjR2':>7} {'resid skew':>11} "
      f"{'y range':>22}  logit CV acc")
for i in range(1, 9):
    X = np.load(f"data/function_{i}/inputs.npy")
    y = np.ravel(np.load(f"data/function_{i}/outputs.npy"))
    n, d = X.shape
    A = np.column_stack([np.ones(n), X])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    ss = 1 - (r**2).sum() / ((y - y.mean())**2).sum()
    adj = 1 - (1 - ss) * (n - 1) / (n - d - 1)
    sk = ((r - r.mean())**3).mean() / (r.std()**3 + 1e-300)
    b = (y > np.median(y)).astype(int)
    k = min(5, b.sum(), n - b.sum())
    acc = (cross_val_score(LogisticRegression(max_iter=5000), X, b, cv=k).mean()
           if k >= 2 else float("nan"))
    print(f"{i:>2} {n:>3} {d:>2} {ss:7.3f} {adj:7.3f} {sk:11.2f} "
          f"{y.min():10.3g}..{y.max():<10.3g}  {acc:.2f}")
