Points acting like support vectors

I fitted an SVM to each function and counted how many readings it needed as support
vectors. Almost all of them. Eighteen out of eighteen on Function 3, thirty seven out
of forty three on Function 8, twelve out of thirteen on Function 1. With this little
data nothing sits comfortably away from the boundary, so every point I have is
propping it up. That says the margin means nothing yet, and that crowding queries into one region will not help.

Function 1 is the exception where two points really do matter. A positive reading at
(0.577, 0.585) and a negative one at (0.650, 0.682) sit either side of a sign change.
That pair is a decision boundary in miniature, and it set this week's query: I stepped only part of the way between them, not past them.

Using a network as a surrogate

I trained a small network per function on a good versus bad split at the median
output, then nudged each input and watched the predicted probability move. That gives
a rough gradient per input.

The results are lopsided. On Function 2 the first input takes 0.89 of the influence.
On Function 1 the second input takes 0.78, on Function 3 the third takes 0.77. These agree with the length scales my Gaussian process already fits. Both say Function 3's first input, Function 7's fourth and Function 8's eighth
do almost nothing. Two unrelated models agreeing makes me willing to stop spending
query precision on those.

Framing it as classification

Splitting each output at its median turns the problem into telling good regions from
bad. It works reasonably. The trade off is what gets lost. A classifier says which
side of a line a point falls, not how sure it is, and misclassifying a good region as
bad means I never look there again. The regions I have not sampled are exactly where
the classifier is least reliable, so the errors land where they do most damage.
Exploration needs uncertainty and classification throws it away.

How well the network actually did

Not that well. It beat logistic regression on Functions 1, 4, 5 and 7, and lost to it
on 3, 6 and 8. An RBF kernel SVM matched or beat it almost everywhere, reaching 0.92
on Function 6 against the network's 0.74.

The more telling result is that not one observed point came back with a predicted
probability near 0.5. The network is confident everywhere, including where it is
wrong. So it did not really approximate a decision boundary. It memorised the points
it was given and drew hard edges around them. Backpropagation was still useful, just
not for that. It is the mechanism that produced the input gradients above, so it told
me which variables matter even though the boundary itself was not worth reading.

Which model I am actually using

The Gaussian process, still. It is the only one of the three that reports how unsure
it is, and that uncertainty is the whole input to my acquisition function. A model that cannot say it does not know about a corner cannot help me decide whether to look there.

On flexibility against interpretability, the network has the most flexibility and the
least interpretability, and with thirteen to forty three points per function that
flexibility mostly buys overfitting. I can read a Gaussian process length scale
directly and know that an input does not matter. Getting the same answer out of the
network took a finite difference loop. The added complexity was not worth it here,
though that would flip if the data ever got large enough to support the extra capacity.

What that means for the queries

Nothing dramatic. Seven of eight functions now pass my leave one out trust check, including Function 5 for the first time, so most queries are exploiting.
Function 3 has failed three weeks running and is still exploring. Function 1 gets a
deliberately small step, because the most it can still gain on that line is about
thirty times, against the thirteen million times it gained last week, and a sign flip
is waiting somewhere in the gap.
