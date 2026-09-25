# Model Card

The model that chooses each week's queries for the black-box optimisation capstone.

## Overview

**Name:** per-function Gaussian process with a measured acquisition switch.

**Type:** Bayesian optimisation. A surrogate model plus an acquisition function, not a
predictor. Its job is to decide where to look next, not to say what is there.

**Version:** 4, as used for round 10. Version history is in the section on how the
strategy evolved, below.

**Input:** the past readings for one function. Each reading is a point with between 2 and
8 coordinates, every coordinate between 0 and 1, paired with the single number the
function returned there. Between 19 and 49 readings per function at present.

**Output:** one new point to query, in the same coordinate space.

## Intended use

Suitable for choosing the next evaluation when evaluations are expensive and few, the
function is not observable, and a rough model plus honest uncertainty is enough to rank
candidate points. That covers hyperparameter tuning, physical experiments and simulation
studies with a hard budget.

Cases to avoid:

- **Do not use it to predict a function value.** It is tuned to rank candidates, not to
  be accurate. On Function 4 its leave-one-out error is about 5 against readings that
  span roughly 10.
- **Do not use it where evaluations are cheap.** With thousands of evaluations available,
  fitting a Gaussian process every round is wasted effort and a direct search will beat it.
- **Do not use it on a function whose behaviour changes across the space** without
  checking first. A stationary kernel assumes the same smoothness everywhere. Function 3
  appears to violate that and has not improved once in nine rounds.
- **Do not trust it outside the region already sampled.** This has cost me two queries
  and thirteen orders of magnitude, and it is the failure mode to watch for.

## Architecture

Gaussian process regression with a Matern 5/2 kernel and a separate length scale per
input, fitted each round by maximising marginal likelihood with ten restarts. A white
noise term is added for Function 2 only, which the brief describes as noisy. Function 5's
outputs are log transformed first.

The next point is chosen by scoring 50,000 random candidates with an acquisition function
and taking the best. Which acquisition function is not fixed. Each round, every function
sits a leave-one-out test: drop a reading, refit, predict it, and compare the error
against simply predicting the average. Pass, and the function uses Expected Improvement
and chases the peak it believes in. Fail, and it uses Upper Confidence Bound, which leans
on uncertainty instead.

Function 1 does not use the Gaussian process at all. Its readings span about 120 orders of
magnitude and change sign, so in double precision the surface reads as flat and there is
no gradient to follow. It is handled by fitting a curve to the magnitude of its readings
along two axes and querying where that curve peaks.

## How the strategy evolved

| Version | Round | What changed and why |
|---|---|---|
| 1 | 1 | One Gaussian process per function, Matern 5/2 with a length scale per input, Expected Improvement for all eight. Six of the eight results then fell outside the model's own confidence band. |
| 2 | 2 | Added the leave-one-out test and the switch between Expected Improvement and Upper Confidence Bound, so the choice between exploiting and exploring is measured rather than guessed. Replaced the Gaussian process on Function 1 with a curve fit. |
| 3 | 8 | Restricted Function 1 to querying between readings already held. Round 7 had extrapolated past them on an unchecked symmetry assumption and fell from 1.18 to 2.2e-13. Round 8 returned 1.98, the best reading of the run. |
| 4 | 9 | Added a trust region on Function 4 after it passed the leave-one-out test and still returned -8.85. Candidates are drawn from a box around its best point. The next reading was -1.26. Box shrunk from 0.15 to 0.10 in round 10. |

## Performance

Best value found per function, against where each started. Nine rounds submitted.

| Function | Inputs | Readings | Best at start | Best now | Rounds since last gain | Leave-one-out error |
|---|---|---|---|---|---|---|
| 1 | 2 | 19 | 7.711e-16 | 1.985 | 1 | not applicable |
| 2 | 2 | 19 | 0.6112 | 0.7679 | 3 | 0.229 |
| 3 | 3 | 24 | -0.03484 | -0.03484 | 9 | 0.082 |
| 4 | 4 | 39 | -4.026 | 0.6312 | 5 | 5.061 |
| 5 | 4 | 29 | 1089 | 7216 | 6 | 1.902 |
| 6 | 5 | 29 | -0.7143 | -0.2411 | 1 | 0.253 |
| 7 | 6 | 39 | 1.365 | 2.191 | 1 | 0.305 |
| 8 | 8 | 49 | 9.598 | 9.952 | 0 | 0.054 |

![Progress by round](../results/progress.png)

Six of the eight have improved on their starting best. Function 1 has moved furthest,
about fifteen orders of magnitude, and every step of that came from replacing the model
rather than adding data.

A second metric matters as much as the best value: whether the model's prediction matched
the result. After round 1, six of eight results fell outside the model's own confidence
band, which is what prompted the leave-one-out test. By round 8 the Function 1 curve
predicted 1.66 and the reading came back at 1.98, its closest prediction of the run.

Improvements per round, counting how many of the eight beat their previous best: 2, 4, 2,
3, 3, 2, 0, 3, 1. The decline is the clearest single summary of how the run has gone.

## Assumptions, constraints and failure modes

**Assumption: the functions are equally smooth everywhere.** That is what a stationary
kernel means. Function 3 has failed the leave-one-out test every week and improved in none
of nine rounds, which is what a violated stationarity assumption looks like.

**Assumption: more data improves the model.** Function 4 disproves it. Its leave-one-out
error rose from about 1.2 at 31 readings to about 5 at 39. Its newer readings come from
regions the kernel cannot reconcile with the rest, so fitting them degrades the fit
everywhere.

**Constraint: sample size.** Between 19 and 49 readings for up to 8 dimensions. Nearly
every reading is a support vector when an SVM is fitted, meaning nothing sits comfortably
away from any boundary.

**Constraint: one query per function per week.** Every choice costs a week, so a wasted
query cannot be recovered by running more of them. This rules out grid search and anything
needing a validation budget.

**Failure mode: confident extrapolation.** The model will happily propose a point far from
anything measured and report a narrow confidence band there. Rounds 3, 4 and 7 all failed
this way, round 7 by thirteen orders of magnitude. The trust region on Function 4 and the
interpolation rule on Function 1 are both patches for this, applied after the fact.

**Failure mode: a leave-one-out test that is too easy to pass.** It compares the model
against a mean predictor, which on a badly behaved function is a low bar. Function 4
passed it and then returned its worst reading of the run.

## Transparency and reproducibility

Everything that chose a query is in the repository, and it is meant to be checkable rather
than taken on trust.

Each round is a script under `notebooks/`, committed with the data as it stood at the time,
so any past round can be checked out and rerun against the readings that were actually
available then. `notebooks/capstone_walkthrough.ipynb` does exactly that: it rolls the data
back by one reading, reruns the leave-one-out test on the rolled-back data, regenerates the
queries, and prints whether they match the file that was submitted. That check is the
reason to believe anything else in this card.

This matters for adaptation as much as for verification. Every change in the version table
above was made because a recorded prediction missed a recorded result. Without the record,
round 7's failure would have been a bad week rather than a reason to stop extrapolating,
and Function 4's rising error would have been invisible.

The strategy was developed with help from Claude and ChatGPT, which is part of what a
reader needs to know to judge it. They were used to think through each round, write the
Python and draft the write-ups. The decisions about what to query were mine, and the
rollback check above exists precisely because a model's suggestion is not evidence. Twice a
model proposed something confident and wrong, and both times it was the recorded prediction
against the recorded result that caught it, not my judgement in the moment.

What a reader would still need in order to fully reproduce this: the eight functions
themselves, which the course does not disclose. Everything on my side is here.

## Trade-offs

**Exploiting requires an honest model, and mine has not always been.** The leave-one-out
test exists because of this, but it is a coarse gate.

**Interpretability over flexibility.** Linear regression, SVMs and a small neural network
were all tested and none replaced the Gaussian process, because none reports uncertainty
and uncertainty is the entire input to the acquisition function. The network in particular
gave no usable decision boundary: not one observed point came back with a predicted
probability near 0.5.

**Robustness over peak performance, latterly.** The trust region on Function 4 has not
raised its best value, which is still 0.63. It cut the worst case from -8.85 to -1.26. With
few rounds left, a bounded loss is worth more than an unlikely gain.
