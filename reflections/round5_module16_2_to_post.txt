Repository structure

The repository is organised by what a file is, not by which week produced it:

```
data/function_1..8/     inputs.npy and outputs.npy, one folder per function
notebooks/              the query generation script for each round, plus side checks
reflections/            the written reflection for each module
submissions/            the exact query strings sent to the portal
results/                figures
docs/                   notes
```

Files inside those folders are named by round and module, so `round5_module16.py`
generates the Round 5 queries and `round5_module16.md` is the reflection that went with
them. That pairing matters. To find why a query was chosen, the script and the reasoning sit under the same name in two folders.

The data folders are the part I would defend hardest. Each round's results go in their own commit, so the repository holds the dataset as it was at every stage, not just as it is now. I can check out any round and rerun that round's script against the
data I actually had. I tested this: Round 8's script, rerun today, reproduces the exact
eight query strings I submitted.

Two things I am changing. The `docs` folder still holds a blank proposal template full of TODO headings from when the repository was set up. It describes nothing about this project and should go. The scripts also duplicate about sixty lines of Gaussian process setup. I kept that on purpose so each round stands alone, but I would factor it out if this ran past thirteen rounds.

Coding libraries and packages

Four do the work: numpy for the data, scipy for optimisation and the normal
distribution that Expected Improvement needs, scikit-learn for the Gaussian process and
the comparison models, and matplotlib for figures.

Scikit-learn is central because it has the specific thing this problem needs, which is a
Gaussian process with a Matern kernel and a separate length scale per input. The
separate length scales are not a detail. They are how I find out which inputs do nothing,
and they have flagged three dead inputs across three functions.

The trade-off I keep returning to is that I have not used PyTorch or TensorFlow, and
this module is about them. I did test a neural network through scikit-learn and it lost
to simpler models on three of the eight functions. The reason is sample size. I have
between 17 and 47 readings per function. A framework built for scaling training solves a problem I do not have. The capacity it offers is capacity I cannot feed.
If the budget were thousands of evaluations instead of thirteen, that answer would
reverse.

Documentation

The README already covers the four things someone needs: what the challenge is and why
that shape of problem matters, what the inputs and outputs look like with worked
examples of the exact submission format, what the objective and constraints are, and how
the approach has developed round by round.

It had fallen behind, so I have updated it. It previously stopped at Round 3. It now
carries Rounds 4 to 8, including the two that went wrong, and a table of the current
best value per function with how many readings each has taken.

Writing up the failures was a deliberate choice. Round 7 lost thirteen orders of
magnitude on Function 1 because I measured one side of a peak and assumed the other side
matched. A README that only lists what worked would hide the most useful thing in the
repository, which is a record of which assumptions turned out to be load-bearing. The
round scripts carry the same honesty: `round7_module18.py` opens with a note explaining
that its central assumption was wrong and pointing at the script that corrects it.

What I would still add is a short results log per round, so the best value per function
over time can be read without opening the data files. At the moment that history lives
in the commits, which is the right place for it but not the easiest place to read.
