# Introduction to machine learning methods

The machine methods we use for SAMueL belong to a methods class known as *supervised machine learning binomial classification*. The models learn to predict a binomial classification (that is there are two alternative possible classification) - in our case, whether a patient is predicted to receive thrombolysis or not. The models learn classification by being provided with a training set of examples, each of which is labelled (received thrombolysis or not). After training, the model is tested on an independent set of examples that is not used for training the model.

## Logistic regression

Logistic Regression is a probabilistic model, meaning it assigns a class probability to each data point. Probabilities are calculated using a logistic function:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

Here, $x$ is a linear combination of the variables of each data point, i.e. $a + bx_1 + cx_2 + ..$, where $x_1$ is the value of one variable, $x_2$ the value of another, etc. The function $f$ maps $x$ to a value between 0 and 1, which may be viewed as a class probability. If the class probability is greater than the decision threshold, the data point is classified as belonging to class 1 (thrombolysis). For probabilities less than the threshold, it is placed in class 0 (no thrombolysis). 

During training, the logistic regression uses the examples in the training data to find the values of the coefficients in $x$ ($a$,$b$,$c$...) that lead to the highest possible accuracy in the training data. The values of these parameters determine the importance of each variable for the classification, and therefore the decision making process. A variable with a larger coefficient (positive or negative) is more important when deciding whether or not to thrombolyse a patient.

## Random Forest 

A random forest is an example of an ensemble algorithm: the outcome (whether or not a patient is thrombolysed) is decided by a majority vote of other algorithms. In the case of a random forest, these 'other algorithms' are decision trees. A random forest is an ensemble of decision trees.

We can think of a decision tree as similar to a flow chart. In {numref}`Figure {number} <tree-fig>`)below we can see that a decision tree is comprised of a set of nodes and branches. The node at the top of the tree is called the *root node* and the nodes at the bottom are *leaf nodes*. Every node in the tree, except for the leaf nodes, splits into two branches leading to two nodes that are further down the tree. A path through the decision tree always starts at the root node. Each step in the path involves moving along a branch to a node lower down the tree. The path ends at a leaf node where there are no further branches to move along.

:::{figure-md} tree-fig
<img src="./decision_tree.gv.png" width="400px">

Example of the structure of a decision tree with nodes (ovals) and branches (arrows). The root node (red) is always at the top of the tree. A path through the tree starts at the root node, moves downwards through internal nodes (yellow) and ends at a leaf node (green). 
:::

The path taken through a tree is determined by the rules associated with each node. The decision tree learns these rules during the training process. The goal of the training process is to find rules for each node such that a leaf node contains samples from one class only: the leaf node a patient ends up in determines the predicted outcome of the decision tree. 

Specifically, given some training data (variables and outcomes), the decision tree algorithm will find the variable that is most descriminative (provides the best seperation of data based on the outcome). This variable will be used for the root node. The rule for the root node consists of this variable and a threshold value. For any data point, if the value of the variable is less than or equal to the threshold value at the root node, the data point will take the left branch and if it is greater than the threshold value it will take the right branch. The process of finding the most descriminative feature and a threshold value is repeated to determine the rules of the internal nodes lower down the tree. Once all data points in a node have the same outcome, that node is a leaf node, representing the end of a path through a tree. Once all paths through the tree end in a leaf node the training process is complete. 

As a random forest is an ensemble of decision trees, during training the algorithm will select a random sample of the training data with replacement and train a decision tree using this sample. This process is repeated many times, the exact number being a parameter of the algorithm corresponding to the number of decision trees in the random forest. 

The resulting random forest is a classifier that can be used to determine whether a data point belongs to class 0 (not thrombolysed) or class 1 (thrombolysed). The path of the data point through every decision tree ends in a leaf node. If there are 100 decision trees in the random forest, and the data point's path ends in a leaf node with class 0 in 30 of the decision trees and a leaf node of class 1 in 70, the random forest takes the majority outcome and classifies the data point as belonging to class 1 (thrombolysed) with a probability of 0.7 (70/100: number of trees voting class 1 / total number of trees).
