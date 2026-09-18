Layers, and where mine came from

I do not stack models, so this prompt did not seem to apply. Then I looked at what I had done to Function 1 and it is hierarchical, I just had not called it that.

There are two levels. The first ignores the sign of every reading and models only how
big it is, which gives a smooth surface I can fit. The second works out where to stand on that surface, which is where the sign and local shape live. Coarse structure first, detail on top. That is stacking, only by hand.

The Gaussian process has no such split. It models the raw value at one level, and on Function 1 that is why it failed: the values run from 1e-124 to 1, so in double precision the surface reads as flat. Separating scale from position made it tractable. I would not have framed it that way without this module.

Leaps against increments

Function 1 has gone from 8.6e-16 to 1.18 across four rounds. That is not an increment,
and it did not come from having more data. It came from throwing the Gaussian process
away and fitting something else.

That is the parallel I take from AlexNet: the jump was not more of ImageNet through the old approach, it was a different architecture on the same data. My Function 3 is the
control case: it has been fed every round's data and has not improved since the very
first submission. More data into the wrong model buys nothing. Eighteen rounds of
increments have produced less than one change of model did.

Trade-offs

Yes, and it is the same trade-off in different clothes. Deciding how deep a network should be is deciding how much flexibility the data supports. My leave one out test
asks that question directly: if a model cannot predict a point it has not seen, it does
not get to chase its own peak, and it explores instead.

Last module's sweep made that hard to ignore. On 43 points one hidden layer beat two, and a heavy penalty cut the run to run spread fivefold. Every change that reduced flexibility helped. Exploiting an overfitted model is the expensive version of this mistake: it costs a query and a week.

Which building block mattered

Gradients, and loss, for different reasons.

Nudging each input and watching the prediction move showed which variables the model is sensitive to, and that agreed with the length scales the Gaussian process fits on its own. Two unrelated methods pointing at the same dead inputs is worth more than
either alone.

Loss I use differently from training. My leave one out error is a validation loss, but it
is a gate rather than something to minimise. It does not adjust any weights. It decides
whether a model has earned the right to be trusted this week.

Prototype or production

Rapid prototyping, clearly, and deliberately. Each round is one self contained script.
Nothing is shared between them, and every round is rerun from scratch.

That would be poor practice if this ran often. It runs thirteen times. The bottleneck is deciding what to query, not computing it, so what matters is being able to change approach completely between rounds, which I have now done twice. The price is duplicated code, which I accept rather than build an abstraction used thirteen times.

What counts as success

The interview made me think about benchmarking rather than scoring. A model in use is judged on whether it holds up on unseen data, not on its best result.

My own benchmark has moved the same way. I used to ask whether the value improved. Now I
ask whether the result matched what the model predicted. Round 5 is the example: the
Function 1 fit predicted log10 of the size at +0.06 and the reading came back at +0.073.
The gain was 2.2 times, which is fine, but the accuracy is what told me the model was
finally right. Two rounds earlier the same fit was wrong by three orders of magnitude
while still improving the value. A number going up is not evidence that you understand
anything.
