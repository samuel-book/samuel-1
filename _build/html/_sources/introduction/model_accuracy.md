# A comparison of machine learning models  
  
The table below provides a summary of key accuracy statistics for the machine learning models.

While the accuracy is better when all data is used to train a single model (using one-hot encoding of hospitals), there is an advantage in simplicity of understanding/communication of having separate models for different hospitals. 
  
| Model\*                                    | Accuracy (%) | ROC-AUC | Max Sens=Spec (%) [1] |
|--------------------------------------------|--------------|---------|-----------------------|
| Logistic regression single model           | 83.2         | 0.904   | 82.0                  |
| Logistic regression hospital-level models  | 77.5         | 0.815   | 74.0                  |
| Random forest single model                 | 84.6         | 0.914   | 83.7                  |
| Random forest hospital-level models        | 81.4         | 0.854   | 78.1                  |
| Fully-connected neural net single model    | 84.4         | 0.913   | 83.3                  |
| Embedding neural net single model          | 85.5         | 0.921   | 84.5                  |

\* Single model fits use one-hot encoding for hospitals. Hospital-level models fit a model to each hospital independently. Embedding neural nets encode hospital id, pathway data, and clinical data into a single value vector each.

[1] The maximum value where sensitivity matches specificity.
