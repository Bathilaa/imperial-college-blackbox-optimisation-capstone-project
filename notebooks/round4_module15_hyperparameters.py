"""
Hyperparameter sweep on Function 8, for the Module 15.2 write-up.
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X = np.load("data/function_8/inputs.npy")
y = np.ravel(np.load("data/function_8/outputs.npy"))
b = (y > np.median(y)).astype(int)
SEEDS = range(5)


def run(**kw):
    base = dict(hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
                learning_rate_init=1e-3, alpha=1e-2, max_iter=3000)
    base.update(kw)
    acc, it = [], []
    for s in SEEDS:
        m = make_pipeline(StandardScaler(), MLPClassifier(random_state=s, **base))
        acc.append(cross_val_score(m, X, b, cv=5).mean())
        it.append(m.fit(X, b)[-1].n_iter_)
    return np.mean(acc), np.std(acc), np.mean(it)


def table(title, label, settings):
    print(f"\n{title}")
    print(f"  {label:>22} {'accuracy':>9} {'spread':>8} {'iters to stop':>14}")
    for name, kw in settings:
        a, s, n = run(**kw)
        print(f"  {name:>22} {a:9.2f} {s:8.3f} {n:14.0f}")


table("LEARNING RATE (continuous, spans orders of magnitude)", "rate",
      [(f"{v:g}", {"learning_rate_init": v}) for v in (1e-4, 1e-3, 1e-2, 1e-1)])

table("NETWORK SIZE (discrete, a count of layers and units)", "hidden layers",
      [("(4,)", {"hidden_layer_sizes": (4,)}),
       ("(16,)", {"hidden_layer_sizes": (16,)}),
       ("(16, 8)", {"hidden_layer_sizes": (16, 8)}),
       ("(64, 32, 16)", {"hidden_layer_sizes": (64, 32, 16)})])

table("L2 PENALTY (continuous, also spans orders of magnitude)", "alpha",
      [(f"{v:g}", {"alpha": v}) for v in (1e-4, 1e-2, 1e-1, 1.0)])

table("OPTIMISER AND ACTIVATION (discrete, unordered choices)", "setting",
      [("adam", {"solver": "adam"}),
       ("sgd", {"solver": "sgd"}),
       ("lbfgs", {"solver": "lbfgs"}),
       ("tanh instead of relu", {"activation": "tanh"}),
       ("logistic activation", {"activation": "logistic"})])
