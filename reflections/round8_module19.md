How I prompted

I used Claude and ChatGPT. No LLM model has chosen one of my query points. Every coordinate
comes out of a script that fits a Gaussian process to my saved readings, scores 50,000
candidates and takes the best, from a fixed seed. Function 1 comes out of a curve fitted to
readings I already hold. What I use the models for is working out what that script should do
next, the harder question.

My prompts are about strategy and structure. I stopped asking open questions early on,
because "what should I query next" gave me answers that were fluent, confident and impossible
to check. Now I write a state summary myself each week: current readings per function, what
my leave-one-out check returned, what I tried and what it cost. I put a proposed change
against it, make it argue the case, and test the argument on my own data before changing any
code. (I also ask for the plain English version, because a claim I can restate simply is one
I can check, and one that cannot is hiding an assumption)

Decoding settings

I worked in chat interfaces, so temperature, top-p and top-k were not exposed and I did not
set them. They would not have touched a coordinate in any case, since those come out of code.
What they would change is the strategy conversation: low when checking a claim against my
numbers, higher when asking what else I could try.

That is the same trade-off I already manage in the optimisation. Expected Improvement is the
low temperature setting and Upper Confidence Bound is the high one. This week I put six
functions on EI and one on UCB, and that split came from my check, not from a model.

Token limits

Precision is the risk. The portal takes each coordinate at exactly six decimal places and
rejects anything else, so one dropped digit costs a week. My scripts format the numbers and
write the submission file, and my readings live in .npy files rather than in prompts, so
nothing I submit has passed through a tokeniser.

(No truncation so far, checked by putting the same reasoning to two conversations and
comparing. At thirteen rounds my summary would near 300 readings, where I would pass file
paths instead)

What went wrong

Round 7 was my mistake. I had a good point on Function 1, probed 0.03 to one side, saw it
cost almost nothing, and concluded the ridge was symmetric. Nothing in my data said that. I
stepped 0.12 the other way and the value fell from 1.18 to 2.2e-13. The failure was an
assumption I made and never tested. I changed my strategy afterwards: Function 1 is now
queried only between points I already hold. Round 8 returned 1.98, the best of the run.

Tackling Hallucinations

First, my rule that nothing counts until a script reproduces it from saved data. The notebook
in my repo rolls my data back a reading, regenerates the round 8 queries and checks them
against the file I submitted. Second, I put the same summary to Claude and ChatGPT and check
wherever they disagree. Third, I read the output properly. I caught one referring to a
function's true optimum, which is not knowable from eighteen readings.

Scaling this up

The main problem would be the amount of information in my weekly summary. If I had more
functions or more results, I would give the model a data file. I would also focus on
automating, making script do more of the checks.

Since I can test each function only once per 'week' (in my case, only once per input). So a
poor choice wastes that week's opportunity. I use the results I already have when I want to
make a safer choice. When I want to explore, I consider areas where I have less information.

For example, in round 7, I explored too far without enough evidence and got a very poor
result. After that, I changed my process so that every important decision could be checked
and repeated using my code. The final decision no longer depended only on whether I trusted
the model's advice or not
