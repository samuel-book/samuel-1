# Neural Networks

This section describes experiments using neural network classifiers to predict whether a patient would, or would not, receive thrombolysis. Data is restricted to admissions within 4 hours of stroke onset (to units that have at least 300 patients, and 10 thrombolysis uses, over three years)

This section contains the following notebooks:

## Fully connected neural network

In these notebooks, we use a fully connected neural network, that is all input features are linked to all neurones in the first layer of the network, and subsequently all neurones of a layer connect to all neurones in the next layer.

* *Fully connected TensorFlow model - Check required training epochs*: Measuring the accuracy of the model against training and test data sets to ascertain the optimal number of training epochs (the number of times the training data is passed through the model to train it).

* *Fully connected TensorFlow model - Train and save models*: Trains 5 models (on 5-fold cross validation train/test data sets) and saves for later use.

* *Fully connected TensorFlow model - analysis*: Analyses the saved models for: 1) Various accuracy scores, 2) Receiver-Operator Characteristic Curve, 3) Sensitivity-Specificity Curve, and 4) model calibration.

* *Fully connected TensorFlow model - Learning curve*: Analysis the learning curve of the fully connected neural net.

## Modular neural networks

Modular neural nets split data into three groups: 1) hospital id, 2) patient/clinical characteristics, 3) pathway times/timings. Each subgroup of data is processed by a neural subnet, the output of which may be a vector of any number of values - here we use 1 and 2 values per subnet output. The output from the subnets is combined in an additional layer in the neural network, and that layer outputs a sigmoid probability of receiving thrombolysis. This is called 'embedding' of the three subgroups of data.

When the subnets output a single value, this will condense each of the subgroups down to a single value that will be used in the final layer to determine probability of thrombolysis. This allows, for example, ranking of patients suitability for thrombolysis determined by a consensus view from all hospitals, and similarly allows ranking of hospitals to be ranked by propensity to give thrombolysis, independent of their own patient population. When 2 or more output values are used for each subnet this allows more complex interactions between patients and hospitals, and offers the potential to cluster similar hospitals or patients by location of their output vectors.

The notebooks are:

* *Modular TensorFlow model with 1D embedding - Train and save models*: Trains 5 models (on 5-fold cross validation train/test data sets) and saves for later use. Each subnet outputs a single value.

* *Modular TensorFlow model with 1D embedding - analyse*: Analyses the saved models for: 1) Various accuracy scores, 2) Receiver-Operator Characteristic Curve, 3) Sensitivity-Specificity Curve, and 4) model calibration.

* *Modular TensorFlow model with 2D embedding - Train and save models*: Trains 5 models (on 5-fold cross validation train/test data sets) and saves for later use. Each subnet outputs a pair of values.

* *Modular TensorFlow model with 2D embedding - analyse*: Analyses the saved models for: 1) Various accuracy scores, 2) Receiver-Operator Characteristic Curve, 3) Sensitivity-Specificity Curve, and 4) model calibration.




