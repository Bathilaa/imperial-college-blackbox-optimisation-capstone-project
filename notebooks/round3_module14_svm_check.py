"""
SVM check for the Module 14 reflection only.
Splits each function's outputs into high and low at the median, then compares a
soft-margin linear SVM against an RBF kernel SVM, plus logistic regression as
last week's baseline. Small data, so cross validated accuracy on the whole set.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

print(f"{'F':>2} {'n':>3} {'d':>2} {'logistic':>9} {'linear SVM':>11} "
      f"{'RBF SVM':>8}   best C for RBF")
for i in range(1, 9):
    X = np.load(f"data/function_{i}/inputs.npy")
    y = np.ravel(np.load(f"data/function_{i}/outputs.npy"))
    b = (y > np.median(y)).astype(int)
    n, d = X.shape
    k = min(5, b.sum(), n - b.sum())
    if k < 2:
        print(f"{i:>2} {n:>3} {d:>2}   too few in one class")
        continue
    logi = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, b, cv=k).mean()
    lin = cross_val_score(make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)), X, b, cv=k).mean()
    best, bestC = -1, None
    for C in [0.1, 1, 10, 100]:
        s = cross_val_score(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale")), X, b, cv=k).mean()
        if s > best:
            best, bestC = s, C
    print(f"{i:>2} {n:>3} {d:>2} {logi:9.2f} {lin:11.2f} {best:8.2f}   C={bestC}")
