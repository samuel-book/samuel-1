# Comparison of average values for patients who receive thrombolysis and those that do not

## Aims

Compare feature means for patients who receive thrombolysis and those that do not.

This analysis is for only those patients arriving within 4 hours of known stroke onset, and is on data that has been coded and, where necessary, imputed.The data used in this analysis is the data used for machine learning.

## Load and analyse data

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up results DataFrame
results = pd.DataFrame()

# Display entire dataframes
pd.set_option("display.max_rows", 999, "display.max_columns", 150)

# Import data (concenate training and test set used in machine learning)
data = pd.concat([
    pd.read_csv('./../data/10k_training_test/cohort_10000_train.csv'),
    pd.read_csv('./../data/10k_training_test/cohort_10000_test.csv')])

# Add columns for scanned within 4 hours of arrival and onset
data['scan_within_4_hrs_arrival'] = data['S2BrainImagingTime_min'] <= 240
data['scan_within_4_hrs_onset'] = \
    (data['S1OnsetToArrival_min'] + data['S2BrainImagingTime_min']) <= 240
# Convert boolean to integer
data['scan_within_4_hrs_arrival'] *= 1
data['scan_within_4_hrs_onset'] *= 1

data.dtypes

## Summarise all 4 hour admissions

all_4hr_admissions_summary = data.describe().T
results['all'] = all_4hr_admissions_summary['mean']
all_4hr_admissions_summary

## Summarise 4 hour admissions who receive thrombolysis

mask = data['S2Thrombolysis'] == 1
thrombolysis_admissions = data[mask]
thrombolysis_admissions_summary = thrombolysis_admissions.describe().T
results['thrombolysis'] = thrombolysis_admissions_summary['mean']
thrombolysis_admissions_summary

## Summarise 4 hour admissions who do not receive thrombolysis

mask = data['S2Thrombolysis'] == 0
no_thrombolysis_admissions = data[mask]
no_thrombolysis_admissions_summary = no_thrombolysis_admissions.describe().T
results['no_thrombolysis'] = no_thrombolysis_admissions_summary['mean']
no_thrombolysis_admissions_summary

## Show summary of all groups

Add ratio of yes/no thrombolysis (and sort by ratio), and save.

results['ratio'] = results['thrombolysis'] / results['no_thrombolysis']
results.sort_values('ratio', inplace=True)
results.to_csv('output/thrombolse_yes_no_means.csv')
results

## Observations

For patients arriving within 4 hours of known stroke onset, compared to those that do not receive thrombolysis, patients who receive thrombolysis:

* Have a confirmed ischaemic stroke 
* Do not have stroke onset during sleep
* Arrive outside of the hours 3am to 6am
* Are younger (mean age 73 vs 76)
* Arrive sooner (mean onset to arrival 97 vs 117 minutes)
* Have higher stroke severity (mean NIHSS 11.6 vs 8.4)
* Are scanned within 4 hours of arrival (100% vs 93%) and 4 hours of onset (99% vs 77%)
* Are more likely to have a precisely determined stroke onset time (97% vs 87%)
* Have arrived by ambulance (94% vs 91%)
* Not have atrial fibrillation (14% vs 24% having AF)
* Not have a history of TIA (21% vs 30% having had TIA)
* Not be on anticoagulant