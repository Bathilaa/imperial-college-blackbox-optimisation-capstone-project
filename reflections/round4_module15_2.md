The sweep

I ran a small network on Function 8, which has the most data, and changed one setting
at a time. Five runs of each, so I could see the spread as well as the average.

Learning rate moved convergence more than anything else. At 0.0001 it took 2,639
rounds to settle and still scored worst at 0.74. At 0.01 it took 89 rounds and scored
0.82. At 0.1 it took 22 rounds and scored about the same. So the low rate was not
being careful. It was just slow, and it stopped somewhere worse.

Network size surprised me. A single layer of 16 units scored 0.85 and was the steadiest
of everything I tried, with a spread of 0.011. My own default of 16 then 8 scored 0.81
with five times that spread. The biggest network, 64 then 32 then 16, got 0.84. Depth
bought nothing here.

The L2 penalty did almost nothing until I turned it up hard. At 1.0 the accuracy rose
a little and the spread dropped from 0.054 to 0.011.

The optimiser mattered more for stability than accuracy. lbfgs settled in 226 rounds
with a spread of 0.012. Adam took 502 rounds with 0.054. Plain SGD took 1,341 and came
last.

The pattern is that everything making the model less flexible made it better. One
layer, a heavy penalty, a solver that heads straight for a minimum. With 43 points
that is not surprising, but I would not have guessed the ranking in advance.

Continuous or discrete

Learning rate and the L2 penalty are continuous, and both matter across orders of
magnitude rather than in even steps. Going from 0.0001 to 0.001 changed more than
going from 0.01 to 0.1. So they belong on a log scale, and they suit any method that
assumes a smooth surface underneath.

Layers, units per layer, choice of optimiser and choice of activation are discrete.
Layers and units are counts, so there is no halfway. Optimiser and activation are not
even ordered. Adam does not sit between SGD and lbfgs on any scale.

The mix is the awkward part. A method that treats layer count as a continuous number
will happily suggest 2.4 layers, which means nothing. So the search has to handle both
kinds at once, log scaling the smooth ones and picking from a list for the rest.

What I will do with it

Two things.

First, if I use a network again it gets one hidden layer and a strong penalty. My
default was worse than the simplest thing I tried, which is a reminder to test
defaults rather than inherit them.

Second, and more useful: tuning hyperparameters is itself a black box problem. I
cannot write down accuracy as a function of learning rate. I can only pick a setting
and see what comes out, and each attempt costs time. That is the same shape as the
capstone, so the same tools should work. I could fit a Gaussian process over the
hyperparameters and use Expected Improvement to choose the next setting instead of
running a grid.

The reverse is also worth admitting. My grid was seventeen runs to learn what a
handful of well chosen ones would have told me. I had the better method sitting in the
same folder and did not use it on my own model. That is the honest lesson from this
week.
