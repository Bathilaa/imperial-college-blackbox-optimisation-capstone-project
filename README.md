# Black-box optimisation capstone

Imperial ML&AI certificate, cohort IMP-PCMLAI-26-02.

Eight hidden functions. One query each per week for 13 weeks. Find the input that
gives the highest output for each. No formula, no gradients, nothing but the number
that comes back.

## Data

All inputs are between 0 and 1. Queries go to the portal hyphen separated, six
decimal places:

```
Function 1: 0.540849-0.536214
Function 3: 0.737112-0.024565-0.001448
```

Output is one number per function. The scales vary a lot.

| Function | Inputs | Readings | Best so far |
|---|---|---|---|
| 1 | 2 | 17 | 1.184 |
| 2 | 2 | 17 | 0.7679 |
| 3 | 3 | 22 | -0.03484 |
| 4 | 4 | 37 | 0.6312 |
| 5 | 4 | 27 | 7216 |
| 6 | 5 | 27 | -0.3246 |
| 7 | 6 | 37 | 1.804 |
| 8 | 8 | 47 | 9.95 |

## Approach

Bayesian optimisation. A Gaussian process per function, Matern kernel, one length
scale per input. Expected Improvement picks the next point.

Each round:

- refit the GP with the new reading
- leave-one-out check: drop a point, refit, predict it. If the model can't beat just
  predicting the average, it uses UCB instead of EI
- score 50,000 random candidate points, take the best

Function 1 doesn't use a GP. Its values run from 1e-124 to 1, so in double precision
the surface reads as flat. I fit a curve to the size of the readings instead and use
that for direction only.

## Rounds

| Round | What changed |
|---|---|
| 1 | GP + EI on all eight |
| 2 | Added the leave-one-out check. Dropped the GP for Function 1 |
| 3 | Tried SVMs |
| 4 | Tried a neural network classifier, swept its hyperparameters |
| 5 | Curve through three points on F1's line. Predicted +0.06, got +0.073 |
| 6 | That line was done, so probed sideways |
| 7 | Stepped 0.12 sideways and fell 13 orders of magnitude. I'd measured one side of the peak and assumed the other matched. Nothing improved on any function |
| 8 | F1 interpolates between readings instead of stepping past them |

## Other models I tried

Linear regression, SVM, neural network. None of them replaced the GP.

Linear regression fits badly, which is expected on curved surfaces. The SVM separates
high from low output regions, but nearly every reading ends up a support vector so the
margin means nothing at this sample size. The neural network beat logistic regression
on four functions and lost on three, and no point came back with a predicted
probability near 0.5, so it drew hard edges rather than a boundary.

None of them report uncertainty, which is the whole input to Expected Improvement.
They stay as diagnostics. The network's input gradients were useful though: they agree
with the GP length scales on which inputs do nothing.

## Layout

```
data/function_1..8/   inputs.npy, outputs.npy
notebooks/            one script per round, plus side checks
reflections/          the write-up for each module
submissions/          the query strings sent to the portal
```

Each round's results are their own commit, so the data at any past round can be
checked out and that round's script rerun against it.

## Running it

```
pip install -r requirements.txt
python notebooks/round8_module19.py
```
