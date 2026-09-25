Week 8: Function 1 up to 1.98 from 1.18, Functions 6 and 7 also improved. Function 4
returned -8.85, its worst yet.

How I prompted

No model has chosen one of my query points. Every coordinate comes out of a script that
fits a Gaussian process to my saved readings, scores 50,000 candidates and takes the best,
from a fixed seed. Function 1 comes out of a curve fitted to readings I already hold. What
I use the models for is working out what that script should do next, the harder question.

My prompts are about strategy, not coordinates. I stopped asking open questions early on,
because "what should I query next" gave me answers that were fluent, confident and
impossible to check. Now I write a state summary myself each week: current readings per
function, what my leave-one-out check returned, what I tried and what it cost. I put a
proposed change against it, make it argue the case, and test the argument on my own data
before changing any code. I also make it explain the idea in plain English, because if I
cannot follow it I cannot defend the decision here.

Decoding settings

I worked in chat interfaces, so temperature, top-p and top-k were not exposed and I did
not set them. They would not have touched a coordinate in any case, since those come out
of code. What they would change is the strategy conversation: low when checking a claim
against my numbers, higher when asking what else I could try.

That is the same trade-off I already manage in the optimisation. Expected Improvement is
the low temperature setting and Upper Confidence Bound is the high one. This week I put
six functions on EI and one on UCB, and that split came from my check, not from a model.

Token limits

The risk here is precision, not length. The portal takes each coordinate at exactly six
decimal places and rejects anything else, so one dropped digit costs a week. My scripts
format the numbers and write the submission file, and my readings live in .npy files
rather than in prompts, so nothing I submit has passed through a tokeniser.

No truncation so far, checked by putting the same reasoning to two conversations and
comparing. At thirteen rounds my summary would near 300 readings, the point where I would
pass file paths instead.

What went wrong

Round 7 was my mistake. I had a good point on Function 1, probed 0.03 to one side, saw it
cost almost nothing, and concluded the ridge was symmetric. Nothing in my data said that.
I stepped 0.12 the other way and the value fell from 1.18 to 2.2e-13. The failure was not
an invented fact from a model but an assumption I made and never tested. I changed my rule
afterwards: Function 1 is now queried only between points I already hold. Round 8 returned
1.98, the best of the run.

Stopping it making things up

Three things, in order of how much they helped. First, my rule that nothing counts until a
script reproduces it from saved data. The notebook in my repo rolls my data back a
reading, regenerates the round 8 queries and checks them against the file I submitted.
Second, I put the same summary to Claude and ChatGPT and check wherever they disagree.
Third, I read the output properly. I caught one referring to a function's true optimum,
which is not knowable from eighteen readings.

Scaling this up

My summary breaks first. With more functions or readings I would pass file paths instead
of numbers, and move more of the weekly reasoning into code so less depends on a
conversation. I would automate my checks too, because reading carefully does not scale.

Working without knowing the answer

This is the capstone problem again. One query per function per week is my whole budget, so
a wrong step costs a week. Grounding every question in my own numbers is exploitation,
asking openly is exploration, and Round 7 is what exploration costs unconstrained. What
helped was not trusting the models less. It was keeping the decisions in code I can rerun,
so trust stopped mattering.
