Week 2 changes

Last week I trusted my models too much. So this week I tested them.

I took each model back to the data I had before Round 1. I asked it to guess the point I went on to query. Then I compared that guess with the answer the portal sent back. Most guesses were well off, and the models had said they were confident. Function 4 is the clearest one. The model guessed about -2. The real answer was a small positive number, and it is now the best reading I have on that function.

That matters because Expected Improvement believes whatever the model says about its own confidence. If the confidence is fake, the query is wasted.

So I added a simple test. For each function I take one point out, refit the model, and ask it to guess the point I removed. If it does better than just guessing the average, I keep Expected Improvement. If not, I switch to Upper Confidence Bound, which puts more weight on what the model does not know. Functions 3 and 5 failed and switched. The rest passed.

I also dropped the Gaussian process for Function 1. I said last week I would. Its readings are tiny and keep flipping between plus and minus, so the surface looks flat to the computer and there is nothing to follow. Instead I ignored the signs, kept the size of each reading, and fitted a simple curve to those. It fits well and points near the middle of the square, so that is where I am querying. I only half believe it. The same curve says the value there will be huge, which cannot be true. I am using the spot it picked and ignoring the rest.

Exploring or exploiting

I did not pick one for the whole round. The test above picked for me, function by function.

Functions 4, 6, 7 and 8 passed, so I let them chase the best spot they know. Functions 2, 3 and 5 either failed or still have very few points, so those queries went somewhere new. Function 1 is neither. It is just a test of my curve.

The risk is that chasing the best spot only works if the model is honest, and this week showed mine is not. But I have eleven rounds left. If I only ever explore, I never collect anything. Letting the test decide felt safer than guessing.

Other people's posts

Chase Bender went the other way. He is exploring harder and moved everything to UCB. Eduardo Wizentier gave a rule: exploit when the predicted best sits more than 1.5 standard deviations above what you already have. I like having a rule instead of a feeling. But my test says my standard deviations are too small to trust, so my rule uses past errors instead. May Zune's regression work is why I ran the regressions below rather than guess at what they would show.

Where regression falls over

I fitted a straight line model to each function. Mostly it did badly. That makes sense. These surfaces bend and a straight line cannot. On Function 1 it does worse than just predicting the average. Function 8 looks good at first, but it only has about five readings per input, so it is probably fitting noise.

The other assumptions fail too. On several functions the errors lean one way instead of spreading evenly. Function 5 runs from under one to over a thousand, so the errors are nowhere near the same size across the space.

I also tried logistic regression. I split each output at its middle value and asked the model to tell high from low. Function 1 is a coin toss. Function 3 does worse than guessing. Function 7 is the odd one out and gets it right about four times in five, so there is a real line between its high and low areas even though the straight line fit was poor. I am not sure how useful that is yet. If it holds, I would use it to rule out areas before spending a query there, not to predict values.
