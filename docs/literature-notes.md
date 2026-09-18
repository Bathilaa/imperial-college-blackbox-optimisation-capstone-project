# Reading notes

Papers behind the choices in this project, and what each one actually changed.

## Jones, Schonlau & Welch (1998) - Efficient Global Optimization of Expensive Black-Box Functions

The origin of Expected Improvement. Fit a surrogate to what you have, then pick the next
point by balancing its predicted value against how uncertain the surrogate is there.

Why it matters here: it is built for exactly my constraint, where each evaluation is
expensive and few are available. It also makes clear why the surrogate must report
uncertainty. A model that only predicts cannot tell you when to go and look somewhere new.

## Snoek, Larochelle & Adams (2012) - Practical Bayesian Optimization of Machine Learning Algorithms

Changed a concrete choice. They argue against the squared exponential kernel because it
assumes the function is unrealistically smooth, and recommend Matern 5/2 with a separate
length scale per input.

I use exactly that. The per-input length scales have since flagged three inputs across
three functions as carrying no information.

## Rasmussen & Williams (2006) - Gaussian Processes for Machine Learning

The reference for the Gaussian process itself. Kernels, the posterior, and fitting
hyperparameters by marginal likelihood, which is what runs every round.

## Eriksson et al. (2019) - TuRBO (Scalable Global Optimization via Local Bayesian Optimization)

Not used yet. Keeps the search inside a trust region that shrinks after failures and
grows after successes, rather than trusting one global model everywhere.

This is the fix for Function 4. Its leave-one-out error rose from 1.19 at 31 readings to
4.74 at 37, and its returned values swung 0.63, -7.35, -0.12, because the global model
keeps proposing points far from anything measured. A trust region would have pulled the
step in after the first failure rather than the third.

## Snoek et al. (2014) - Input Warping for Bayesian Optimization of Non-Stationary Functions

Also not used yet. Handles surfaces whose behaviour changes across the space.

This is my best explanation for Function 3, which has not improved in eighteen rounds
and fails the leave-one-out test every week. A stationary kernel assumes the same
smoothness everywhere. If that assumption is wrong, no amount of data fixes it.

## Shahriari et al. (2016) - Taking the Human Out of the Loop

Survey of Bayesian optimisation. The place to look for acquisition functions I have not
tried.

## Benchmarks

COCO and BBOB provide standard black-box test functions with known optima. Useful
because an acquisition function can be tested there as often as I like, instead of
spending a week of real queries finding out it does not help.
