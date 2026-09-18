Building up in layers

A CNN gets edges before it gets objects, and it cannot skip a level. My Function 1 work
has gone the same way without me planning it.

Three levels so far. First, where is there any signal at all, which meant modelling the
size of the readings and ignoring their sign. Second, which direction the signal runs.
Third, where exactly to stand on that line. Each one only became answerable once the one
below it was settled.

Round 6 is the clearest case. The curve through my line fitted to within 0.01 and its
peak landed on the point I had already queried. There was nothing left to learn at that
resolution. So I moved up a level and probed sideways, which is a feature I had never
measured. The line was finished, in the same way a layer stops contributing once it has
extracted what it can.

Breakthroughs and grinding

LeNet looks like one breakthrough but it was convolution, pooling, backpropagation and
weight sharing arriving together, each useless without the others.

My Function 1 gains look like leaps. It has moved about fifteen orders of magnitude.
But the curve fit that produced them only worked because earlier rounds had established
roughly where the signal lived. Drop that scaffolding and the fit has nothing to fit.

Function 3 is the counter example, and it is the one that bothers me. Its best reading
is still the fourth one I ever took, eighteen rounds ago. It has failed my trust check
every single week. There has been no scaffolding to build on, so there has been no leap
either, and more data has not manufactured one.

Depth, cost and overfitting

Round 6 was exactly this trade-off. My curve already fitted the line almost perfectly. I
could have spent the query refining it further, which is adding depth to a model that
already fits the training points. That is overfitting in the plainest sense: improving
the fit without improving what I know.

The cost side is sharper here than in training a network. One query is one week. Fitting
an already solved thing more precisely does not just waste compute, it costs a round I
cannot get back. So I spent it on a direction I had no data for at all.

One building block

Pooling. A pooling layer deliberately throws away precision, keeping only what survives
at a coarser scale.

My length scales do the same thing in reverse. When an input's length scale pins at the
ceiling, the model is telling me that dimension carries nothing, and I can stop spending
precision on it. Three inputs across three functions sit at that ceiling now.

Convolution gave me the more uncomfortable thought. Weight sharing means the same filter
applies everywhere in the image, which is fine when the thing you are looking for looks
the same wherever it appears. My kernel makes the same assumption: the same smoothness
everywhere in the space. If Function 3 behaves differently in different regions, that
assumption is wrong, and no amount of data will fix a model that cannot represent the
shape. That is a better explanation of eighteen flat rounds than bad luck.

Deployment and benchmarks

The edge AI discussion was about accepting the best model that fits the constraint rather
than the best model. My constraint is thirteen queries per function. Everything follows
from that.

So the benchmark I care about is not whether a query improved the best value. It is
whether the query bought information worth a week. Round 6's sideways probe came back
lower than my best, 1.11 against 1.18. Judged on value it failed. Judged on information
it was the most useful query of the round: it told me the perpendicular direction was
nearly flat, which four rounds of walking the line had never revealed.

Both of my Function 1 mistakes came from optimising the wrong benchmark. Chasing value
and treating the information as a bonus, rather than the other way round.
