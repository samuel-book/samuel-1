# Introduction to machine learning methods

## Logistic regression

Logistic Regression is a probabilistic model, meaning it assigns a class probability to each data point. Probabilities are calculated using a logistic function:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

Here, $x$ is a linear combination of the variables of each data point, i.e. $a + bx_1 + cx_2 + ..$, where $x_1$ is the value of one variable, $x_2$ the value of another, etc. The function $f$ maps $x$ to a value between 0 and 1, which may be viewed as a class probability. If the class probability is greater than the decision threshold, the data point is classified as belonging to class 1 (thrombolysis). For probabilities less than the threshold, it is placed in class 0 (no thrombolysis). 

During training, the logistic regression uses the examples in the training data to find the values of the coefficients in $x$ ($a$,$b$,$c$...) that lead to the highest possible accuracy in the training data. The values of these parameters determine the importance of each variable for the classification, and therefore the decision making process. A variable with a larger coefficient (positive or negative) is more important when deciding whether or not to thrombolyse a patient.
