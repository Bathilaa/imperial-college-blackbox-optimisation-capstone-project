Repository structure

The repository is organised by what a file is, not by which week produced it:

```
data/function_1..8/   inputs.npy and outputs.npy
notebooks/            query generation script per round, plus side checks
reflections/          one per module
submissions/          the exact query strings sent to the portal
results/ docs/        figures, notes
```

Files are named by round and module, so `round5_module16.py` generates the Round 5 queries and `round5_module16.md` is the reflection that went with them. To find why a query was chosen, the script and the reasoning sit under the same name in two folders.

The data folders are the part I would defend hardest. Each round's results go in their own commit, so the repository holds the dataset as it stood at every stage, not only as it is now.

Two things I am changing. The `docs` folder still holds a blank proposal template full of TODO headings from when the repository was set up; it describes nothing about this project and should go. The scripts also duplicate about sixty lines of setup. I kept that so each round stands alone, but would factor it out if this ran past thirteen rounds.

Coding libraries and packages

Four do the work: numpy for the data, scipy for optimisation and the normal
distribution that Expected Improvement needs, scikit-learn for the Gaussian process and
the comparison models, and matplotlib for figures.

Scikit-learn is central because it has the thing this problem needs: a Gaussian process with a Matern kernel and a separate length scale per input. Those length scales are not a detail. They are how I find which inputs do nothing, and they have flagged three across three functions.

The trade-off I keep returning to is that I have not used PyTorch or TensorFlow, and this module is about them. I did test a neural network through scikit-learn and it lost to simpler models on three of eight. The reason is sample size. I have
between 17 and 47 readings per function. A framework built for scaling training solves a problem I do not have. The capacity it offers is capacity I cannot feed.
If the budget were thousands of evaluations instead of thirteen, that answer would
reverse.

Documentation

The README covers four things: what the challenge is and why that shape of problem matters, what the inputs and outputs look like with worked examples of the submission format, the objective and its constraints, and how the approach developed round by round.

It had fallen behind, so I have updated it. It previously stopped at Round 3. It now
carries Rounds 4 to 8, including the two that went wrong, and a table of the current
best value per function with how many readings each has taken.

Writing up the failures was deliberate. Round 7 lost thirteen orders of magnitude on Function 1 because I measured one side of a peak and assumed the other matched. The scripts carry the same honesty: `round7_module18.py` opens with a note saying its central assumption was wrong and pointing at the script that corrects it.

Who the repository is for

Two audiences, wanting different things.

A collaborator needs to rerun a decision. They can check out Round 5 and rerun its script against the data I held that week, and get the queries I submitted. I tested that on Round 8 today and it reproduces exactly. Reproducibility that has not been tried is a claim, not a property.

An employer reads for judgement rather than output. They cannot verify the answers and would not want to, so what is legible is how decisions were made and what happened when they were wrong. A repository of only successes shows someone who either never took a risk or edited the record. Neither reads well.

What I would still add is a short results log per round, so the best value per function
over time can be read without opening the data files. At the moment that history lives
in the commits, which is the right place for it but not the easiest place to read.
