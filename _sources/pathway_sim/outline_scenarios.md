# Testing of alternative what-if? scenarios

This section describes experiments using the pathway simulation. Key componenents of each hospital's pathway may be changed, and the effect on thrombolysis use and clinical benefit estimated. Sceanrios tested are:

1. Base: Uses the hospitals' recorded pathway statistics in SSNAP (same as validation notebook). 

2. Speed: Sets 95% of patients having a scan within 4 hours of arrival, and all patients have 15 minutes arrival to scan and 15 minutes scan to needle. 

3. Onset-known: Sets the proportion of patients with a known onset time of stroke to the national upper quartile if currently less than the national upper quartile (leave any greater than the upper national quartile at their current level). 

4. Benchmark: The benchmark thrombolysis rate takes the likelihood to give thrombolysis for patients scanned within 4 hours of onset from the majority vote of the 30 hospitals with the highest predicted thrombolysis use in a standard 10k cohort set of patients. These are from Random Forests models. 

5. Combinations of the above.

This section contains the following notebooks:

* *Stroke pathway simulation - generation of results from alternative sceanrios*: Generate results for alternative sceanrios (no analysis)

* *Analysis of alternative pathway scenarios*: Numerical and graphicla analysis of results from the scenario modelling.
