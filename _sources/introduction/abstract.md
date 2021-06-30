# Abstract

**Background**: Stroke is a common cause of adult disability. Expert opinion is that about 20% of patients should receive thrombolysis to break up a clot causing the stroke. Currently 11-12% of patients in England and Wales receive this treatment, ranging from 2% to 25% between hospitals.

**Objectives**: Enhance national stroke audit by understanding of the key sources of inter-hospital variation and how a target of 20% thrombolysis may be reached.

**Design**: Modelling, using machine learning and clinical pathway simulation. Qualitative research on clinician engagement with models.

**Participants & Data Source**: Anonymised data for 246,676 emergency stroke admissions to acute stroke teams in England and Wales between 2016 and 2018, obtained from the Sentinel Stroke National Audit data (SSNAP).

**Results**: Clinical decision making can be predicted with 85% accuracy for those patients with a chance of receiving thrombolysis (arriving within four hours of stroke onset). Machine learning models allow prediction of likely treatment choice of each patient at all hospitals. Clinical pathway simulation predicts hospital thrombolysis use with an average absolute error of 0.5 percentage points. Three changes were applied to all hospitals in the model: 1) arrival-to-treatment in 30 minutes, 2) proportion of patients with determined stroke onset times set at least the national upper quartile, 3) thrombolysis decisions made based on majority vote of a benchmark set of 30 hospitals. Any single change alone is predicted to increase thrombolysis use from 11.6% to 12.3% to 14.5% (clinical decision-making have most effect). Combined, these changes would be expected to increase thrombolysis to 18.3% (and double clinical benefit of thrombolysis as speed increases also improve clinical benefit independently of the proportion of patients receiving thrombolysis), but there would still be significant variation between hospitals depending on local patient population. For each hospital the effect of each change may be predicted, alone or in combination. Qualitative research showed that engagement with, and trust in, the model was greatest in physicians from units with higher thrombolysis rates. Physicians also wanted to see a machine learning model predicting outcome with probability of adverse effect of thrombolysis, to counter a fear that driving thrombolysis use up may cause more harm than good.

**Limitations**: Models may only be built using data available in SSNAP. Not all factors affecting use of thrombolysis are contained in SSNAP data; the model therefore provides information on patterns of thrombolysis use in hospitals, but is not suitable for, or intended for, a decision aid to thrombolysis.

**Conclusions**: Machine Learning and clinical pathway simulation may be applied at scale to national audit data, allowing extended use and analysis of audit data. Stroke thrombolysis rates of at least 18% look achievable in England and Wales, but each hospital should have its own target.

**Future work**: Extent machine learning modelling to outcome and probability of adverse effects of thrombolysis. Co-production of communication of model outputs with stroke units.

**Funding**: This project was funded by the National Institute for Health Research (NIHR) HS&DR programme and will be published in full in NIHR Health Services and Delivery Research Journal.

