Week 3 changes

Round 2 went better than Round 1. Four of eight improved, up from two. The big one was Function 1.

Last week I dropped the Gaussian process for Function 1 and fitted a simple curve to the sizes of the readings instead. It picked a spot. The reading there came back positive at 8.6e-12, about eleven thousand times bigger than my previous best positive.

But the curve was also badly wrong. It said the value there would be around 1e8. The real one was 1e-11, nineteen orders of magnitude out. So the curve is good at pointing and useless at predicting. I use it for direction only now.

I worked out two options for Function 1 this week and picked one.

Option A was to refit the curve with the new point and query its new peak. The peak barely moved, about 0.03. Small step, small gain.

Option B uses the fact that my biggest reading anywhere sits at (0.65, 0.68) but is negative, while my best positive reading is at (0.54, 0.54). So I step a third of the way from the positive one toward the big one. That should gain about three orders of magnitude if the sign holds.

I went with B. The nearest sign flip I know about is roughly 0.1 across and my step is 0.06, so it should stay positive. If it flips, I learn where the boundary is, which is worth having.

On tuning: mostly I do not. I let the Gaussian process fit its own length scales and leave the rest alone. The one thing I choose is which acquisition function each function gets, and a test picks that, not me.

Exploring or exploiting

Same rule as last week. I take one point out, refit, and see whether the model predicts it better than just guessing the average. Pass means it has earned the right to exploit. Fail means explore.

Functions 4, 6 and 7 passed and are exploiting. Functions 3 and 5 failed again, second week running, so they keep exploring. Functions 2 and 8 passed but their queries still landed in uncertain ground.

Functions 3 and 5 failing twice is what I keep noticing. Another week of data did not fix them. That points at the model being wrong for those two, not a shortage of data.

Would SVMs help

I tried it rather than guess. I split each function's outputs into high and low at the middle value, then fitted a soft margin SVM to tell them apart. I ran a straight line version and an RBF kernel version, scaling the inputs first as SVMs need.

The kernel version wins where you would expect. Function 7 goes from 0.82 to 0.88, Function 4 from 0.63 to 0.81, Function 2 from 0.57 to 0.73. Those are the bendy ones. On Functions 3, 5 and 6 the straight line version is better, and on Function 8 they are level at 0.76.

So a soft margin SVM can tell good regions from bad on most of these, and the kernel helps where the surface bends. But I would not let it pick a query. An SVM gives me a decision boundary and tells me which side of it a point sits on. It does not tell me how unsure it is, and that uncertainty is the whole reason Expected Improvement works. I would use it as a filter: throw out the low half, then let the Gaussian process choose in what is left. On Function 1 that is worth something: the kernel version gets 0.67 while the straight line version manages 0.47, worse than a coin toss.

What is breaking as the data grows

Three things.

Overfitting is real and I can see it. My leave one out test is exactly a check for it, and two functions fail every week.

Some inputs do nothing. The model fits a length scale per input, and a very long one means that input barely matters. Function 3's first input, Function 7's second and Function 8's last are all pinned at the ceiling. I still specify them to six decimal places, which is wasted.

Dimensions hurt. Function 8 has 42 points across 8 inputs, about five per input. Nothing I fit there is really trustworthy, which is probably why its SVM score sits at 0.76 whichever kernel I use.

What this teaches

This is a stripped down version of a lot of real work. You cannot see what you are optimising, every attempt costs time or money, and the model you decide with is itself uncertain and sometimes badly wrong. Tuning a model where each run takes hours has the same shape.

The thing I keep relearning is that being wrong is fine as long as you find out quickly. My Function 1 curve was wrong by nineteen orders of magnitude and still the most useful model I had, because it was wrong in a way I could measure. The models to worry about are the ones I never test.

The other habit is checking instead of assuming. Last week I ran the regressions rather than guess. This week I ran the SVMs. Both times the answer was more mixed than I expected. Working with incomplete information comes down to that: make a guess, write down what you expect, then go and find out.
