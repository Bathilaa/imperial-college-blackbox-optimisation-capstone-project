Which hyperparameters, and why those

Most of mine are not hand set. The Gaussian process fits its own each round by maximising marginal likelihood: the length scales, the signal variance, and on Function 2 the noise level. I choose the structure, the data chooses the numbers.

The one I prioritised is giving every input its own length scale rather than sharing
one. That means more parameters fitted from very little data, which is a real price. I
pay it because those length scales are the only thing telling me which inputs do
nothing. Three inputs across three functions now sit pinned at their ceiling, which is
the model saying that dimension carries no information, and I have stopped spending
precision on them.

The two I do set myself are the kernel, Matern rather than squared exponential because
the latter assumes a smoothness these functions do not have, and the exploration weight
on UCB.

How tuning changed my queries

Early rounds used one setup everywhere and trusted whatever came out. Now the fitted hyperparameters decide the query, not only the prediction.

The mechanism is my leave one out test. Drop a reading, refit the model and its
hyperparameters without it, predict the missing point. If that beats predicting the
average, the function keeps Expected Improvement and chases its peak. If not, it moves
to UCB and explores instead.

Function 3 has failed that test every single week and has been exploring for six rounds.
Everything else currently passes. So tuning shifted one function from refinement to
broad search, and the decision was measured rather than chosen.

Methods, and what they cost

Three, none of them a grid.

Marginal likelihood fitting does most of the work, with ten restarts because one optimisation lands in local optima. That costs about eight seconds per function, nothing against a week per evaluation. It is principled, but on small data it can overfit confidently.

Manual adjustment covers what the data cannot decide: the log transform on Function 5,
the noise term on Function 2 alone because the brief calls it noisy. Cheap and
transparent, but it is my judgement rather than evidence.

Grid search I used once, on the neural network in Module 15, where 43 points allowed cross validation. On the actual functions it is impossible: one real evaluation per function per week leaves no validation budget.

What 16 points exposes

More data has made one of my models worse, which I did not expect.

I recomputed Function 4's leave one out error at each data size. At 31 points it was
1.19. At 37 points it is 4.74. Four times worse for six extra readings. Its returned
values over the same stretch went 0.63, then -7.35, then -0.12.

Function 8 did the opposite across the same growth: 0.138 at 41 points, 0.093 at 47.

So volume is not the variable. Function 4's newer readings come from regions the kernel cannot reconcile with the rest, so fitting them degrades the fit everywhere. That is capacity, not shortage. It is the strongest argument I have for a trust
region, which would stop the model extrapolating into ground it cannot represent.

Where I would take this

Adaptive rather than fixed. A trust region radius is itself a hyperparameter that tunes
during the run, shrinking after failures and growing after successes. Function 4 would
have had its step pulled in after the first bad reading rather than the third.

The other direction I noticed last module. Tuning is a black box problem in its own right: expensive to evaluate, no gradient. I ran a seventeen run grid to learn what a handful of well chosen settings would have told me. Using my own method on my own model is the obvious next step.

What it is teaching me

To decide under a budget I cannot extend, with information I know is incomplete. The
question I keep returning to is not whether a number went up but whether the model had
earned the right to be believed. I have got that wrong twice, and both times it cost a
week.
