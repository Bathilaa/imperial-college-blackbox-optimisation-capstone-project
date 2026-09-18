# Datasheet

For the data behind the black-box optimisation capstone.

## Motivation

The dataset exists to run a structured optimisation challenge. Eight functions are
hidden from the participant, who submits one input per function per week and receives
one number back. The point is to practise optimising something you cannot see inside,
which is the situation in hyperparameter tuning, physical experiments and any setting
where each trial is expensive.

It was created by Imperial College Executive Education for the Professional Certificate
in Machine Learning and Artificial Intelligence. The functions themselves, their forms
and their true optima, are not disclosed to participants.

## Composition

Each instance is one query and its result: a point in the input space, and the single
number the function returned there.

- Eight functions, with 2, 2, 3, 4, 4, 5, 6 and 8 inputs respectively.
- Every input is a real number between 0 and 1.
- 175 readings were supplied at the start, from 10 for Function 1 up to 40 for
  Function 8.
- 231 readings after seven rounds. Every round adds exactly eight, one per function.

There is no missing data. Every query submitted has returned a value.

Nothing here is confidential or personal. The inputs are coordinates and the outputs
are numbers from a synthetic function. No individual is represented.

## Collection process

The starting readings were provided by the course. Everything since has been generated
by me, one round per week, by choosing a point and submitting it through the capstone
portal. Results arrive by email one to three days later.

This is not a sample of anything larger. The points are chosen, not drawn, and they are
chosen deliberately: mostly by fitting a Gaussian process to what I already have and
picking the point that maximises Expected Improvement.

Collected from late August 2026 onward, one round per week.

## Preprocessing, cleaning and labelling

Very little, and all of it reversible.

- Function 5's outputs are log transformed before modelling, because they span from
  about 0.1 to over 7000. The stored data is raw.
- Function 1 is modelled on the magnitude of its readings rather than the signed value,
  because it takes both signs and spans about 120 orders of magnitude. Again, the
  stored data is raw.
- Nothing is discretised, bucketed or removed.

The raw values are what sit in the repository. Every transform happens at model time
inside the round scripts, so the stored data never loses information.

## Uses

The obvious other use is as a benchmark. Because the functions are fixed and the
readings are honest, the same data can test a different acquisition function or
surrogate model offline, without spending real queries.

One thing a consumer needs to know: **this data is not a random sample of the input
space, and it is heavily biased towards regions that looked promising.** Fitting a
global model to it and claiming it describes the whole function would be wrong. Large
areas have never been visited, and the places that have been sampled most densely are
precisely the places my search already believed were good.

It should not be used to estimate how these functions behave on average, or to claim
anything about their global structure.

## Distribution

The starting data was distributed by the course to enrolled participants. My readings
are in this public GitHub repository.

I do not know the licence terms for the underlying functions. The course materials do
not state them, so I have published only the query and response values, which are my
own submissions and their results, and not any course material describing the
functions.

## Maintenance

Me, for the duration of the capstone. Each round's results are committed separately, so
the dataset as it stood at any past round can be recovered from the history.
