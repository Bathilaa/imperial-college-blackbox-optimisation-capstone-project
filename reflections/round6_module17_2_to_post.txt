Technical justification

My approach is Gaussian process Bayesian optimisation. One query per function per week,
thirteen in total, so every evaluation is expensive and none can be wasted. That is the
exact setting the method was built for.

Jones, Schonlau and Welch (1998) set this out in their Efficient Global Optimization
paper. Fit a surrogate to what you have, then use Expected Improvement to pick the next
point, balancing the predicted value against how uncertain the surrogate is there. The
reason it fits my problem is that it needs both numbers, and a Gaussian process gives
both at once. A model that only predicts cannot tell me when to go and look somewhere
new.

Papers I have actually used

Snoek, Larochelle and Adams (2012) is the one that changed a concrete choice. They argue
against the squared exponential kernel for practical problems because it assumes the
function is unrealistically smooth, and recommend Matern 5/2 with a separate length
scale per input. I use exactly that. The per input length scales have since flagged
three inputs across three functions as carrying nothing.

Rasmussen and Williams (2006) is the reference for the Gaussian process itself.

The one that would strengthen the project is Eriksson et al. (2019), TuRBO. It keeps the
search inside a trust region that shrinks after failures and grows after successes,
instead of trusting one global model everywhere. My Function 4 is the case for it: it
has swung from 0.63 to -7.3 to -0.12 in three rounds, because the global model keeps
proposing points far from anything I have measured and being confidently wrong. A trust
region would have pulled those queries back in after the first failure.

Snoek et al. (2014) on input warping is the other one I want. It handles surfaces whose
behaviour changes across the space, which is my best explanation for Function 3 refusing
to improve for eighteen rounds while failing my trust check every week.

Libraries and why these ones

numpy for the data, scipy for the optimiser and the normal distribution Expected
Improvement needs, scikit-learn for the Gaussian process and for the comparison models.

Scikit-learn over BoTorch or GPyOpt is a deliberate choice, not a default. What I need
is a Matern kernel with per input length scales, and scikit-learn has it. BoTorch would
give me batch acquisition functions, TuRBO and GPU training, none of which addresses my
actual bottleneck. I have between 17 and 47 readings per function and one decision a
week. Nothing there is compute bound.

The honest trade-off is that if I do adopt trust regions, scikit-learn will not carry it
and I would move to BoTorch. I have avoided PyTorch and TensorFlow for the same reason:
capacity I cannot feed with this much data.

Documenting this in GitHub

The README currently says what I do but not what it rests on. I have added a references
section naming each paper against the choice it justifies, so the Matern kernel links to
Snoek and Expected Improvement links to Jones, rather than both looking like defaults I
never questioned.

Each round's script already opens with what changed and why, including the two rounds
that went wrong. Round 7's script says outright that its central assumption was
incorrect and points at the script that corrects it. For anyone reading the repository,
knowing which decisions were reasoned and which were mistakes is more useful than a
clean list of things that worked.

Where to look next

Shahriari et al. (2016) is a survey of the whole field and the sensible place to find
acquisition functions I have not tried.

For benchmarks, the COCO and BBOB suites provide standard black box test functions with
known optima. I could test an acquisition function there for free, as often as I like,
instead of spending a week of real queries finding out it does not help.

The BoTorch documentation is the practical route into trust regions and batch methods if
I decide the extra dependency is worth it.
