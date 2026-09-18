"""
Neural net vs logistic vs SVM on a high/low split, plus input gradients.
For the Module 15 write-up.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

SEED = 0


def setup(i):
    X = np.load(f"data/function_{i}/inputs.npy")
    y = np.ravel(np.load(f"data/function_{i}/outputs.npy"))
    return X, (y > np.median(y)).astype(int)


def nn():
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(16, 8), activation="relu", solver="lbfgs",
        alpha=1e-2, max_iter=5000, random_state=SEED))


print(f"{'F':>2} {'n':>3} {'d':>2} {'logistic':>9} {'RBF SVM':>8} {'neural net':>11} "
      f"{'SVs':>5}  boundary points")
grads = {}
for i in range(1, 9):
    X, b = setup(i)
    n, d = X.shape
    k = min(5, b.sum(), n - b.sum())
    if k < 2:
        print(f"{i:>2} {n:>3} {d:>2}   too few in one class"); continue

    lg = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, b, cv=k).mean()
    sv = cross_val_score(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)), X, b, cv=k).mean()
    nnacc = cross_val_score(nn(), X, b, cv=k).mean()

    # how many observations the SVM needs as support vectors
    s = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)).fit(X, b)
    n_sv = s[-1].support_.size

    # observations the network is genuinely unsure about: the boundary itself
    m = nn().fit(X, b)
    p = m.predict_proba(X)[:, 1]
    near = np.where(np.abs(p - 0.5) < 0.15)[0]

    # which inputs move the network's prediction most (central differences)
    h, g = 0.02, np.zeros(d)
    for kk in range(d):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, kk] = np.clip(X[:, kk] + h, 0, 1); Xm[:, kk] = np.clip(X[:, kk] - h, 0, 1)
        g[kk] = np.abs(m.predict_proba(Xp)[:, 1] - m.predict_proba(Xm)[:, 1]).mean()
    grads[i] = g / (g.sum() + 1e-12)

    print(f"{i:>2} {n:>3} {d:>2} {lg:9.2f} {sv:8.2f} {nnacc:11.2f} {n_sv:5d}  {len(near)} of {n}")

print("\nrelative input influence on the network's prediction (shares, largest first)")
for i, g in grads.items():
    order = np.argsort(-g)
    top = ", ".join(f"x{k+1} {g[k]:.2f}" for k in order[:3])
    print(f"  F{i}: {top}")
