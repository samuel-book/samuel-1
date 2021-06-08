# Assessing accuracy of models with straified k-fold cross-validation

When assessing accuracy of the machine learning models, stratified k-fold splits were used. We used 5-fold splits where each data point is in one, and only one, of five test sets. This is represented schematically below. Data is stratified such that the test set is representative of hospital mix in the whole data population, and within the hospital level data the use of thrombolysis is representative of the whole data for that hospital.

![](../images/kfold.jpg)
