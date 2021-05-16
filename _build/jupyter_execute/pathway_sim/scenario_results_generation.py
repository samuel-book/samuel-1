# Stroke pathway simulation - generation of results from alternative scenarios

This notebook runs alternative pathway simulations, with adjusted pathway parameters. The scenarios are:

1) Base: Uses the hospitals' recorded pathway statistics in SSNAP (same as validation notebook)

2) Speed: Sets 95% of patients having a scan within 4 hours of arrival, and all patients have 15 minutes arrival to scan and 15 minutes scan to needle.

3) Onset-known: Sets the proportion of patients with a known onset time of stroke to the national upper quartile if currently less than the national upper quartile (leave any greater than the upper national quartile at their current level).

4) Benchmark: The benchmark thrombolysis rate takes the likelihood to give thrombolysis for patients scanned within 4 hours of onset from the majority vote of the 30 hospitals with the highest predicted thrombolysis use in a standard 10k cohort set of patients. These are from Random Forests models.

5) Combine *Speed* and *Onset-known*

6) Combine *Speed* and *Benchmark*

7) Combine *Onset-known* and *Benchmark*

8) Combine *Speed* and *Onset-known* and *Benchmark*

Results are saved for each hospital and scenario. Detailed analysis will be performed in subsequent notebooks

## Import libraries and data

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathway import model_ssnap_pathway_scenarios

# load data from csv and store in pandas dataframe
filename = './hosp_performance_output/hospital_performance.csv'
hospital_performance_original = pd.read_csv(filename, index_col=0)

## Get results of alternative scenarios

### Base scenario

The base scenario uses the hospitals' recorded pathway statistics in SSNAP (same as validation notebook)

results_all = model_ssnap_pathway_scenarios(hospital_performance_original)
results_all['scenario'] = 'base'

# Save pathway stats used
hospital_performance_original.to_csv('output/performance_base.csv')

### Speed (30 minute arrival to needle)

The adjusted speed scenario sets 95% of patients having a scan within 4 hours of arrival, and all patients have 15 minutes arrival to scan and 15 minutes scan to needle.

# Create scenarios
hospital_performance = hospital_performance_original.copy()
hospital_performance['scan_within_4_hrs'] = 0.95
hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
hospital_performance['scan_needle_mins_mu'] = np.log(15)
hospital_performance['scan_needle_mins_sigma'] = 0

# Get results
results = model_ssnap_pathway_scenarios(hospital_performance)
results['scenario'] = 'speed'

# Add to results_all
results_all = pd.concat([results_all, results], axis=0)

# Save pathway stats used
hospital_performance.to_csv('output/performance_speed.csv')

### Known onset

Set the proportion of patients with a known onset time of stroke to the national upper quartile if currently less than the national upper quartile (leave any greater than the upper national quarile at their current level).

# Create scenarios
hospital_performance = hospital_performance_original.copy()

onset_known = hospital_performance_original['onset_known']
onset_known_upper_q = np.percentile(onset_known, 75)
adjusted_onset_known = []
for val in onset_known:
    if val > onset_known_upper_q:
        adjusted_onset_known.append(val)
    else:
        adjusted_onset_known.append(onset_known_upper_q)
hospital_performance['onset_known'] = adjusted_onset_known

# Get results
results = model_ssnap_pathway_scenarios(hospital_performance)
results['scenario'] = 'onset'

# Add to results_all
results_all = pd.concat([results_all, results], axis=0)

# Save pathway stats used
hospital_performance.to_csv('output/performance_onset.csv')

### Use benchmark thrombolysis

The benchmark thrombolysis rate takes the likelihood to give thrombolysis for patients scanned within 4 hours of onset from the majority vote of the 30 hospitals with the highest predicted thrombolysis use in a standard 10k cohort set of patients. These are from Random Forests models.

See Random Forests notebooks: *Benchmark hospitals* and *How would thrombolysis use change if clinical decisions were made by hospitals with the highest current thrombolysis rate?* 

# Load benchmark rates
filename = './hosp_performance_output/benchmark_4hr_scan.csv'
benchmark = pd.read_csv(filename, index_col=0)
# Convert from percentage to fraction
benchmark *= 0.01

# Merge in benchmark rates (to ensure order is correct)
hospital_performance = hospital_performance_original.copy()
hospital_performance = hospital_performance.merge(
    benchmark, left_index=True, right_index=True, how='left')
hospital_performance['eligable'] = hospital_performance['benchmark']

# Get results
results = model_ssnap_pathway_scenarios(hospital_performance)
results['scenario'] = 'benchmark'

# Add to results_all
results_all = pd.concat([results_all, results], axis=0)

# Save pathway stats used
hospital_performance.to_csv('output/performance_benchmark.csv')

### Combine speed and onset known

* 95% patients have scan within 4 hours of arrival

* 30 min arrival to needle (15 minute arrival-to-scan, 15 minute scan-to-needle)

* Proportion of patients with known stroke onset time set to upper quartile if currently lower

# Create scenarios
hospital_performance = hospital_performance_original.copy()

# Speed
hospital_performance['scan_within_4_hrs'] = 0.95
hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
hospital_performance['scan_needle_mins_mu'] = np.log(15)
hospital_performance['scan_needle_mins_sigma'] = 0

# Onset known
onset_known = hospital_performance_original['onset_known']
onset_known_upper_q = np.percentile(onset_known, 75)
adjusted_onset_known = []
for val in onset_known:
    if val > onset_known_upper_q:
        adjusted_onset_known.append(val)
    else:
        adjusted_onset_known.append(onset_known_upper_q)
hospital_performance['onset_known'] = adjusted_onset_known

# Save pathway stats used
hospital_performance.to_csv('output/performance_speed_onset.csv')

# Get results
results = model_ssnap_pathway_scenarios(hospital_performance)
results['scenario'] = 'speed_onset'

# Add to results_all
results_all = pd.concat([results_all, results], axis=0)

## Combine speed and benchmark

* 95% patients have scan within 4 hours of arrival

* 30 min arrival to needle (15 minute arrival-to-scan, 15 minute scan-to-needle)

* Decison to thrombolyse patients if scanned within 4 hours of known onset as predicted from majority vote of 30 benchmarl hospitals

# Create scenarios
hospital_performance = hospital_performance_original.copy()

# Speed
hospital_performance['scan_within_4_hrs'] = 0.95
hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
hospital_performance['scan_needle_mins_mu'] = np.log(15)
hospital_performance['scan_needle_mins_sigma'] = 0

# Benchmark
hospital_performance = hospital_performance.merge(
    benchmark, left_index=True, right_index=True, how='left')
hospital_performance['eligable'] = hospital_performance['benchmark']

# Get results
results = model_ssnap_pathway_scenarios(hospital_performance)
results['scenario'] = 'speed_benchmark'

# Add to results_all
results_all = pd.concat([results_all, results], axis=0)

# Save pathway stats used
hospital_performance.to_csv('output/performance_speed_benchmark.csv')

## Combine onset-known and benchmark

* Proportion of patients with known stroke onset time set to upper quartile if currently lower

* Decison to thrombolyse patients if scanned within 4 hours of known onset as predicted from majority vote of 30 benchmarl hospitals

# Create scenarios
hospital_performance = hospital_performance_original.copy()

# Onset known
onset_known = hospital_performance_original['onset_known']
onset_known_upper_q = np.percentile(onset_known, 75)
adjusted_onset_known = []
for val in onset_known:
    if val > onset_known_upper_q:
        adjusted_onset_known.append(val)
    else:
        adjusted_onset_known.append(onset_known_upper_q)
hospital_performance['onset_known'] = adjusted_onset_known

# Benchmark
hospital_performance = hospital_performance.merge(
    benchmark, left_index=True, right_index=True, how='left')
hospital_performance['eligable'] = hospital_performance['benchmark']

# Get results
results = model_ssnap_pathway_scenarios(hospital_performance)
results['scenario'] = 'onset_benchmark'

# Add to results_all
results_all = pd.concat([results_all, results], axis=0)

# Save pathway stats used
hospital_performance.to_csv('output/performance_onset_benchmark.csv')

## Combine speed, onset-known, and benchmark

* 95% patients have scan within 4 hours of arrival

* 30 min arrival to needle (15 minute arrival-to-scan, 15 minute scan-to-needle)

* Proportion of patients with known stroke onset time set to upper quartile if currently lower

* Decison to thrombolyse patients if scanned within 4 hours of known onset as predicted from majority vote of 30 benchmarl hospitals

# Create scenarios
hospital_performance = hospital_performance_original.copy()

# Speed
hospital_performance['scan_within_4_hrs'] = 0.95
hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
hospital_performance['scan_needle_mins_mu'] = np.log(15)
hospital_performance['scan_needle_mins_sigma'] = 0

# Onset known
onset_known = hospital_performance_original['onset_known']
onset_known_upper_q = np.percentile(onset_known, 75)
adjusted_onset_known = []
for val in onset_known:
    if val > onset_known_upper_q:
        adjusted_onset_known.append(val)
    else:
        adjusted_onset_known.append(onset_known_upper_q)
hospital_performance['onset_known'] = adjusted_onset_known

# Benchmark
hospital_performance = hospital_performance.merge(
    benchmark, left_index=True, right_index=True, how='left')
hospital_performance['eligable'] = hospital_performance['benchmark']

# Get results
results = model_ssnap_pathway_scenarios(hospital_performance)
results['scenario'] = 'speed_onset_benchmark'

# Add to results_all
results_all = pd.concat([results_all, results], axis=0)

# Save pathway stats used
hospital_performance.to_csv('output/performance_speed_onset_benchmark.csv')

results_all

## Save results

results_all['stroke_team'] = results_all.index
results_all.to_csv('./output/scenario_results.csv', index=False)