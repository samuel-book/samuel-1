# Random Forests

This section describes experiments using a Random Forest classifier to predict whether a patient would, or would not, receive thrombolysis. Data is restricted to admissions within 4 hours of stroke onset (to units that have at least 300 patients, and 10 thrombolysis uses, over three years)

This section contains the following notebooks:

* *Random Forest Classifier - Fitting to all stroke teams together*: A Random Forest classifier that is fitted to all data together, with each stroke-team being a one-hot encoded feature. Analyses the models for: 1) Various accuracy scores, 2) Receiver-Operator Characteristic Curve, 3) Sensitivity-Specificity Curve, 4) model calibration, and 5) Learning rate.

* *Random Forest Classifier - Fitting hospital-specific models*:  A Random Forest classifier that has a fitted model for each hospital. Analyses the models for: 1) Various accuracy scores, 2) Receiver-Operator Characteristic Curve, 3) Sensitivity-Specificity Curve, and 4) model calibration.

* *Train and save hospital-level Random Forest models*: Train and save hospital-level models for future use (saves models along with thresholds that give expected thrombolysis use at own hospital, and data used to train model. These are saved as a pickled dictionary).

* *Compare level of agreement in clinical decision making between hospitals using Random Forests models*: Pass all patients through all hospital decision models. What level of agreement is there between hospitals?

* *Benchmark hospitals*: Identify top 30 thrombolysis use hospitals (using the same 10k reference set of patients at all hospitals). For each hospital, predict decision made by those benchmark hospitals and take majority decision as whether a patient should be given thrombolysis. Compare these benchmark thrombolysis rates with actual thrombolysis rates.

* *Grouping hospitals by similarities in decision making*: We compare the extent of similarity in decision-making between hospitals, and identify groups of hospitals making similar decisions.

* *Patient level decisions at own hospital, and consensus decisions at other other hospitals*: Predicts and records expected thrombolysis decision at own hospital, and whether majority of benchmark hospitals would give thrombolysis, or 80%/90% of all other hospitals would give thrombolysis.

* *Thrombolysis hospital vs. benchmark diagrams and patient vignettes*: Illustrating difference in decision-making between hospitals and the benchmark set - how much overlap is there, and how much difference? Creating synthetic patient vignettes (based on SSNAP data) to illustrate examples of differences in decision making, e.g. hospitals with lower-than-usual use of thrombolysis in either patients with milder strokes, or patients with prior disability.

* *Find similar patients who are treated differently within the same hospital*: Within a hospital identify patients where the model misclassified the patients. Then look for the most similar patients who where treated differently.

* *What characterises a patient who is predicted to have high variation in thrombolysis use across different hospitals?*: Explotrs the differences between patients predicted to receive thrombolysis at at least 70% of hospitals, and those that are predicted to receive thrombolysis at at 40 to 70% of hospitals.
