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

**Regressions and SVMs.** Linear regression fits badly on most of these, expected for
curved surfaces. A soft-margin SVM does separate high-output from low-output regions, and
an RBF kernel beats a linear one where the surface bends. Neither gives uncertainty
though, which is the whole reason Expected Improvement works, so I keep them as
diagnostics rather than query pickers.

**Exploration versus exploitation.** I do not choose one for the whole round; the
leave-one-out test decides per function. The trade-off is real either way: exploiting
only pays if the model is honest, and mine has not always been, while exploring
everywhere means never collecting on what I have learned. Letting out-of-sample error
decide is more defensible than picking a side by instinct.
