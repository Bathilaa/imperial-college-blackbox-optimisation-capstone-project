# Black-box optimisation capstone

## Project overview

There are eight hidden functions and I cannot see any of them. Each week I submit one
input point per function and a few days later get one number back for each. That is
the only feedback. Over thirteen rounds I must find the input giving the highest
output.

No formula, no gradients, no promise the surface is smooth or has one peak. The only
decision available is where to look next.

This shape turns up all over real machine learning: tuning hyperparameters where each
run takes hours, a physical experiment, optimising anything where one attempt
costs real time or money. You cannot try everything, so the question becomes which
single attempt is worth making.

## Inputs and outputs

Every input is between 0 and 1. The count differs per function.

| Function | Inputs | Starting points |
|---|---|---|
| 1 | 2 | 10 |
| 2 | 2 | 10 |
| 3 | 3 | 15 |
| 4 | 4 | 30 |
| 5 | 4 | 20 |
| 6 | 5 | 20 |
| 7 | 6 | 30 |
| 8 | 8 | 40 |

Queries go into the portal as hyphen-separated values to six decimal places:

```
Function 1: 0.540849-0.536214
Function 3: 0.737112-0.024565-0.001448
```

The output is one number per function and the scales differ wildly. Function 5 returns
values from 0.1 to over 1000. Function 1 returns values near 1e-12 and smaller, with a
flipping sign:

```
Function 1: 8.612454121504655e-12
```

## Objectives and constraints

The goal is to maximise every function. The constraints shape everything:

- One query per function per week over thirteen rounds.
- Results come back a day or two later, so a wasted query costs a week.
- The functions are unknown. No gradients, no structure to rely on.
- Starting data is thin, worst in high dimensions: Function 8 has 40 points across 8
  inputs.

## Technical approach

The core method is Bayesian optimisation. I fit a Gaussian process, which gives a
prediction and a measure of how unsure it is, then use an acquisition function to pick
the next point. I use a Matern kernel rather than RBF, which assumes a very smooth
surface and gets overconfident where there is no data. Each input gets its own length
scale, so the model shows which ones matter.

**Round 1.** Gaussian process plus Expected Improvement on all eight. Log transform on
Function 5, a noise term only on Function 2.

**Round 2.** Six of eight Round 1 results fell outside the model's own confidence
range. So I added a leave-one-out test: drop a point, refit, predict it, compare
against guessing the average. Fail means it does not get to exploit. I also
dropped the Gaussian process for Function 1, whose values sit so close to zero the
surface looks flat, fitting a curve to the sizes of the readings instead.

**Round 3.** Kept the test and ran SVMs to see if they helped.

**Round 4.** Trained a small neural network per function as a classifier on a high or
low output split, and swept its hyperparameters. Function 1's curve located the signal
but was nineteen orders of magnitude out on size, so from here it is used for direction
only.

**Round 5.** Three readings on one line now bracketed Function 1's peak, so a parabola
through them replaced the step-and-hope approach. It predicted log10 of the size at
+0.06 and the reading came back at +0.073, the first accurate call on that function.

**Round 6.** That line was exhausted, so this round probed perpendicular to it.

**Round 7.** The perpendicular probe had cost almost nothing, which looked like a ridge
running across the line. A 0.12 step along it fell thirteen orders of magnitude. The
probe and the step were on opposite sides of the peak and the surface is not symmetric,
so this was an extrapolation dressed up as a ridge. Nothing improved on any function
that round.

**Round 8.** Three readings now bracket that axis too, so the query interpolates between
them rather than stepping past anything.

**Regressions, SVMs and neural networks.** All three were tested and none replaced the
Gaussian process. Linear regression fits badly on curved surfaces. A soft-margin SVM
separates high from low output regions and an RBF kernel beats a linear one where the
surface bends, but almost every reading is a support vector, so the margin means nothing
at this sample size. A small neural network beat logistic regression on four functions
and lost on three, and no observed point came back with a predicted probability near
0.5, meaning it drew hard edges around the points it was given rather than a boundary.
None of the three reports uncertainty, which is the whole input to Expected Improvement,
so all three stay as diagnostics rather than query pickers. The network's input
gradients were the one useful output: they agree with the Gaussian process length scales
on which inputs do nothing.

**Exploration versus exploitation.** I do not choose one for the whole round; the
leave-one-out test decides per function. The trade-off is real either way: exploiting
only pays if the model is honest, and mine has not always been, while exploring
everywhere means never collecting on what I have learned. Letting out-of-sample error
decide is more defensible than picking a side by instinct.

## Where it stands after seven rounds

| Function | Readings | Best value | Found at reading |
|---|---|---|---|
| 1 | 17 | 1.18381 | 15 |
| 2 | 17 | 0.767861 | 16 |
| 3 | 22 | -0.0348353 | 4 |
| 4 | 37 | 0.631155 | 34 |
| 5 | 27 | 7215.67 | 23 |
| 6 | 27 | -0.324613 | 25 |
| 7 | 37 | 1.80354 | 36 |
| 8 | 47 | 9.94968 | 42 |

Function 1 has moved furthest, from 8.6e-16 to 1.18, and every step of that came from
abandoning the Gaussian process for it rather than from collecting more data. Function 3
is the opposite case: it has failed the leave-one-out check every single round and its
best reading is still the one it started with.
