# Round 1 reflection (Module 12)

## What guided each query

I used Bayesian optimisation for all eight functions. For each one I fit a Gaussian process to the data I have, then used Expected Improvement to pick the next query point. I chose EI because it sorts out the exploration vs exploitation trade-off on its own. A point scores well either because the model predicts a high value there, or because the model is unsure about it.

I used a Matern kernel rather than RBF, because RBF assumes the function is very smooth and makes the model too confident where it has no data. I gave each input its own length scale so the model can work out which inputs matter. To find where EI is highest I scored 50,000 random points and took the best.

Two functions needed a transform first. Function 5 runs from 0.11 up to 1089 and Function 1 sits at 1e-16 and below, so I took logs of both. Function 2 is the only one the brief calls noisy, so it is the only one I gave a noise term.

I also checked whether each query was driven by the prediction or by the uncertainty, comparing the model's uncertainty at my chosen point against the median across all candidates. It split four and four:

- Function 1: exploration, but only locally. More below.
- Function 2: exploitation. Ten points in 2D is decent cover, so I trusted the model.
- Function 3: exploration. Only 15 points in 3D, and EI found more value in an area I had not sampled.
- Function 4: exploitation. 30 points in 4D is my best coverage, so I chased the predicted peak.
- Function 5: exploration.
- Function 6: exploration.
- Function 7: exploitation.
- Function 8: exploration. 40 points across 8 inputs is very thin, so uncertainty won.

## The function I found most difficult

Function 1, easily. Every positive reading so far is 7.7e-16 or smaller and some are below 1e-100, so as far as the computer is concerned the surface is flat at zero. The Gaussian process has no gradient to follow, and EI on a flat surface just picks whichever corner has the widest error bars. The one reading that is a decent size, -0.0036, is negative, so I cannot use it once I take logs.

Instead I limited the search to within 0.05 of the strongest positive reading, so at least the query is a sensible local probe. I still think it will come back as basically zero.

Knowing the shape of the signal would have helped. The brief says only proximity gives a non-zero reading, which sounds like a single source fading with distance, but it does not say how fast. A rough idea of the width, or a few more starting points near the one reading that is not tiny, would make this a fitting problem I could solve.

## How I will adapt in future rounds

1. Drop the Gaussian process for Function 1. If it is one source fading with distance, taking logs turns it into a quadratic, which I can fit with least squares and find the peak of directly.

2. Test each model before I trust it. I will leave one point out, refit, predict the missing point, and compare the error against just predicting the mean. If a model cannot beat the mean then its peaks are probably made up, and I will use UCB there instead of EI, since UCB still gives the prediction a say.

3. Look at the fitted length scales. A very long length scale means the model thinks that input does not matter, so I should not waste query precision on it.

I will also note whether each round's result beat what the model predicted, as the easiest check on whether the models are improving.
