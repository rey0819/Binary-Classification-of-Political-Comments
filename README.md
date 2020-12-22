# Binary-Classification-of-Political-Comments
This project trains and tests three AI models (a decision tree, a multinomial naive Bayes, and a multilayer perceptron neural network) using comments made on democratic and republican subreddits.

In order to be able to run the main code (subreddit_project.py), you must first run csvscript.py to generate the needed datasets that are used to train and test the models. We warn that csvscript.py can take a long time to run because of the amount of data it is collecting from the web, in order to ensure we have a good sample size. This sample size can be changed within the code to make the datasets be created more quickly, but will result in less accurate data due to smaller sample size of comments.
