# Black-box optimisation capstone

Imperial ML&AI certificate, cohort IMP-PCMLAI-26-02.

## Non-technical explanation

There are eight hidden machines. Each one takes a few numbers between 0 and 1 and gives
back a single score, and nobody tells you what is inside. My job is to find the settings
that make each score as high as possible.

I get one attempt per machine per week, thirteen in total, so guessing is expensive. So
I build a statistical model of what each machine probably does, use it to pick the
attempt most likely to either score well or teach me something, and update it when the
result comes back a few days later.

The same problem turns up whenever each experiment costs real time or money.

## Data

Eight functions. Every input is between 0 and 1; the number of inputs differs per
function. The output is a single number, and the scales vary enormously.

| Function | Inputs | Readings | Best so far |
|---|---|---|---|
| 1 | 2 | 17 | 1.184 |
| 2 | 2 | 17 | 0.7679 |
| 3 | 3 | 22 | -0.03484 |
| 4 | 4 | 37 | 0.6312 |
| 5 | 4 | 27 | 7216 |
| 6 | 5 | 27 | -0.3246 |
| 7 | 6 | 37 | 1.804 |
| 8 | 8 | 47 | 9.95 |

Queries go to the portal hyphen separated, six decimal places:

```
Function 1: 0.540849-0.536214
Function 3: 0.737112-0.024565-0.001448
```

175 readings were supplied at the start. Each round adds eight, one per function.
Full details in [docs/datasheet.md](docs/datasheet.md).

## Model

Gaussian process regression, Matern 5/2 kernel, one length scale per input. Expected
Improvement picks the next point from 50,000 random candidates.

Each round every function sits a leave-one-out test: drop a reading, refit, predict it,
compare against just predicting the average. Pass and it keeps Expected Improvement and
chases its peak. Fail and it switches to Upper Confidence Bound and explores instead.

Function 1 does not use a Gaussian process. Its values run from 1e-124 to 1 and change
sign, so the surface reads as flat in double precision. A curve fitted to the magnitude
of its readings handles it instead.

Linear regression, SVMs and a small neural network were all tested and rejected. None
of them reports uncertainty, which is the whole input to Expected Improvement.

Full details in [docs/model_card.md](docs/model_card.md).

## Hyperparameter optimisation

Most of them are not set by hand. The Gaussian process fits its own each round by
maximising marginal likelihood with ten restarts: the length scales, the signal
variance, and on Function 2 the noise level.

The per-input length scales earn their cost. They are the only thing that says which
inputs do nothing, and three inputs across three functions now sit pinned at their
ceiling.

Set by hand: the Matern kernel over the squared exponential, the exploration weight on
UCB, the log transform on Function 5 and the noise term on Function 2.

Grid search was used once, on the neural network, where 43 points allowed cross
validation. On the real functions there is no validation budget, because one evaluation
per function per week is the entire supply.

## Results

![Best value found so far, by round](results/progress.png)

Seven rounds in, six of the eight have improved on their starting best.

Function 1 moved furthest, about fifteen orders of magnitude, and every step came from
changing the model rather than collecting more data. Function 3 is the opposite: its
best reading is still the fourth one ever taken, and it has failed the leave-one-out
test every single week.

Two rounds went badly and both are written up rather than hidden. Round 7 stepped
0.12 sideways on Function 1 expecting a flat ridge and fell thirteen orders of
magnitude, because the probe that suggested the ridge was on the other side of the peak
and the surface is not symmetric.

## Layout

```
data/function_1..8/   inputs.npy, outputs.npy
notebooks/            one script per round, plus side checks
reflections/          the write-up for each module
submissions/          the query strings sent to the portal
results/              figures
docs/                 datasheet and model card
```

Each round's results are their own commit, so the data at any past round can be checked
out and that round's script rerun against it.

## Running it

```
pip install -r requirements.txt
python notebooks/round8_module19.py
```

## References

The choices above are not defaults. Each rests on something:

- **Expected Improvement** - Jones, Schonlau & Welch (1998), *Efficient Global
  Optimization of Expensive Black-Box Functions*.
- **Matern kernel with a length scale per input** - Snoek, Larochelle & Adams (2012),
  *Practical Bayesian Optimization of Machine Learning Algorithms*. They argue the
  squared exponential assumes unrealistic smoothness and recommend Matern 5/2 with ARD.
- **Gaussian processes generally** - Rasmussen & Williams (2006), *Gaussian Processes
  for Machine Learning*.

Not used yet, but relevant:

- Eriksson et al. (2019), *TuRBO*. Trust regions that shrink after failures. Function 4
  keeps swinging wildly because the global model proposes far-off points; this is the fix.
- Snoek et al. (2014), *Input Warping for Bayesian Optimization of Non-Stationary
  Functions*. Likely relevant to Function 3.
- Shahriari et al. (2016), *Taking the Human Out of the Loop*. Survey of the field.
