# Benchmark hospitals

## Aims

* Predict thrombolysis decisions for a 10k cohort of patients at all hospitals
* Select top 30 hospitals (highest redicted thrombolysis use in 10k cohort). This is the 'benchmark' set of hospitals.
* Predict decision of those benchmark set of hospitals for all patients at each hopsital. Use a majority vote to classify as whether the patient would receive thrombolysis or not.
* Compare actual thrombolysis use with thrombolysis use if decisions made by majority vote of benchmark hospitals.

# Turn warnings off to keep notebook tidy
import warnings
warnings.filterwarnings("ignore")

## Import libraries 

# Turn warnings off to keep notebook tidy
import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestClassifier

## Load pre-trained hospital models into dictionary *hospital2model* 

keys = hospitals

values = trained_classifier, threshold, patients, outcomes

with open ('./models/trained_hospital_models.pkl', 'rb') as f:
    
    hospital2model = pkl.load(f)

## Import 10k cohort

10k cohort all arrive at hospital within 4 hours of known stroke onset

cohort = pd.read_csv('../data/10k_training_test/cohort_10000_test.csv')

## Pass cohort through all hospital models

Allow for new analysis or used loaded resulkts from previous analysis.

hospitals = list(set(cohort['StrokeTeam'].values))

# Set re_analyse to run analysis again, else loads saved data
re_analyse = False

if re_analyse:
    
    # Get decisions for 10k patients at each hospital
    
    y_test = cohort['S2Thrombolysis']
    X_test =  cohort.drop(['StrokeTeam', 'S2Thrombolysis'], axis=1)   

    results = pd.DataFrame()

    for hospital_train in hospitals:

        model, threshold, _, _ = hospital2model[hospital_train]

        y_prob = model.predict_proba(X_test)[:,1]

        y_pred = [1 if p >= threshold else 0 for p in y_prob]

        results[hospital_train] = y_pred
        
    results.to_csv('./predictions/10k_decisions.csv', index=False)
    
else:
    results = pd.read_csv('./predictions/10k_decisions.csv')

results

### Summarise results

hospital_compare = pd.DataFrame()

hospital_compare['hospital'] = hospitals

hospital_compare['true_rate'] = [
    sum(hospital2model[h][-1])*100/len(hospital2model[h][-1]) for h in hospitals]

hospital_compare['cohort_rate'] = [
    sum(results[h].values)*100/10000 for h in hospitals]

hospital_compare['ratio'] = \
    hospital_compare['cohort_rate'] / hospital_compare['true_rate']

# Label top 30 corhort thrombolysis as benchmark
hospital_compare.sort_values('cohort_rate', inplace=True, ascending=False)
top_30 = [True for x in range(30)]
top_30.extend([False for x in range(len(hospitals)-30)])
hospital_compare['benchmark'] = top_30

hospital_compare

fig = plt.figure(figsize=(6,6))

# Plot copmparison of True rate vs. Corhort rate
ax1 = fig.add_subplot()


colour = ['r'] * 30
colour.extend(['b'] * (len(hospitals) -30 ))


ax1.scatter(hospital_compare['true_rate'],
            hospital_compare['cohort_rate'],
            color=colour,
            label='True vs cohort comparison')

# Add 1:1 line
xx = np.arange(0,50)
ax1.plot(xx,xx, 'k:', label = '1:1 line')

# Add benchmark threshold
mask = hospital_compare['benchmark']
threshold = hospital_compare[mask]['cohort_rate'].min()

ax1.plot([0, 50], [threshold, threshold], 'r--', label='Benchmark threshold')

ax1.set_xlabel('True thrombolysis rate (%)')
ax1.set_ylabel('Predicted cohort thrombolysis rate (%)')
ax1.set_title('')

ax1.legend(loc='lower right')

plt.tight_layout(pad=2)
plt.savefig('./output/benchmark_cohort_with_threshold.jpg', dpi=300)
plt.show()

## Take majority decision of benchmark hospitals  

benchmark_hospitals = hospital_compare['hospital'][0:30]
np.save('./models/benchmark_hospitals.npy', benchmark_hospitals)

# Set re_analyse to run analysis again, else loads saved data
re_analyse = False

if re_analyse:

    columns = \
        np.concatenate((['Hospital','True', 'Majority'], benchmark_hospitals)) 

    results_30 = pd.DataFrame(columns = columns)


    for i, hospital in enumerate(hospitals):
        
        # Show progress
        print(f'Hospital {i+1} of {len(hospitals)}', end='\r')

        # Get hospital model
        _, _, X, y = hospital2model[hospital]

        # Get hospital level results
        hospital_results = pd.DataFrame(columns = columns)           

        # Loop through benchmark hospitals
        for j, top_hospital in enumerate(benchmark_hospitals):
            model, threshold, _, _ = hospital2model[top_hospital]
            y_prob = model.predict_proba(X)[:,1]
            y_pred = [1 if p >= threshold else 0 for p in y_prob]
            hospital_results[top_hospital] = y_pred
        
        # Add to results
        hospital_results['Hospital'] = [hospital for person in y]
        hospital_results['True'] = y
        results_30 = results_30.append(hospital_results, ignore_index=True)
                  
    # Add majority decsion
    majority_threshold = 15/30
    for index,row in results_30.iterrows():
        no = sum([1 for val in row[3:].values if val ==0])
        yes = sum(row[3:].values)
        if yes/(no+yes)>=majority_threshold:
            results_30.loc[index, 'Majority'] = 1
        else:
            results_30.loc[index, 'Majority'] = 0

    results_30.to_csv('./predictions/benchmark_decisions.csv', index=False)
    
else:
    # Load previous results
    results_30 = pd.read_csv('./predictions/benchmark_decisions.csv')

results_30.head()

### Repeat analysis just for patients scanned within 4 hours of onset

This value is used in the pathway simulation model.

# Set re_analyse to run analysis again, else loads saved data
re_analyse = False

if re_analyse:

    columns = \
        np.concatenate((['Hospital','True', 'Majority'], benchmark_hospitals)) 

    results_30_4_hr_scan = pd.DataFrame(columns = columns)


    for i, hospital in enumerate(hospitals):
        
        # Show progress
        print(f'Hospital {i+1} of {len(hospitals)}', end='\r')

        # Get hospital model
        _, _, X, y = hospital2model[hospital]
        
        # Limit to those scanned in 4 hours from onset
        onset_scan = X[:, 1] + X[:, 19]
        mask = onset_scan <= 240
        X = X[mask]
        y = y[mask]

        # Get hospital level results
        hospital_results = pd.DataFrame(columns = columns)           

        # Loop through benchmark hospitals
        for j, top_hospital in enumerate(benchmark_hospitals):
            model, threshold, _, _ = hospital2model[top_hospital]
            y_prob = model.predict_proba(X)[:,1]
            y_pred = [1 if p >= threshold else 0 for p in y_prob]
            hospital_results[top_hospital] = y_pred
        
        # Add to results
        hospital_results['Hospital'] = [hospital for person in y]
        hospital_results['True'] = y
        results_30_4_hr_scan = \
            results_30_4_hr_scan.append(hospital_results, ignore_index=True)
                  
    # Add majority decsion
    majority_threshold = 15/30
    for index,row in results_30_4_hr_scan.iterrows():
        no = sum([1 for val in row[3:].values if val ==0])
        yes = sum(row[3:].values)
        if yes/(no+yes)>=majority_threshold:
            results_30_4_hr_scan.loc[index, 'Majority'] = 1
        else:
            results_30_4_hr_scan.loc[index, 'Majority'] = 0

    results_30_4_hr_scan.to_csv(
        './predictions/benchmark_decisions_4_hr_scan.csv', index=False)
    
else:
    # Load previous results
    results_30_4_hr_scan = pd.read_csv(
        './predictions/benchmark_decisions_4_hr_scan.csv')

results_30_4_hr_scan.head()

### Create summary pivot table

df_pivot = results_30[['Hospital', 'True', 'Majority']].groupby('Hospital')
all_4hr_arrivals = df_pivot.sum() / df_pivot.count()
all_4hr_arrivals['count'] = df_pivot.count()['True']

df_pivot = results_30_4_hr_scan[['Hospital', 'True', 'Majority']].groupby('Hospital')
scan_4hrs_pivot = df_pivot.sum() / df_pivot.count()
scan_4hrs_pivot['count'] = df_pivot.count()['True']

hospital_benchmark_rates = pd.DataFrame()
hospital_benchmark_rates['count_4hr_arrival'] = all_4hr_arrivals['count']
hospital_benchmark_rates['actual_4hr_arrival'] = all_4hr_arrivals['True']
hospital_benchmark_rates['benchmark_4hr_arrival'] = all_4hr_arrivals['Majority']
hospital_benchmark_rates['count_4hr_scan'] = scan_4hrs_pivot['count']
hospital_benchmark_rates['actual_4hr_scan'] = scan_4hrs_pivot['True']
hospital_benchmark_rates['benchmark_4hr_scan'] = scan_4hrs_pivot['Majority']

# Add in whether hospital is in benchmark
benchmark_list = hospital_compare.set_index('hospital')
hospital_benchmark_rates= pd.concat([hospital_benchmark_rates, benchmark_list['benchmark']],axis=1)

hospital_benchmark_rates.to_csv('predictions/hospital_benchmark_rates.csv')
hospital_benchmark_rates

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()

# Plot non-benchmark hospitals in blue
mask = hospital_benchmark_rates['benchmark'] == False
non_bench = hospital_benchmark_rates[mask]

for i, val in non_bench.iterrows():
    ax.plot([non_bench['actual_4hr_arrival'] * 100,
             non_bench['actual_4hr_arrival'] * 100],
            [non_bench['actual_4hr_arrival'] * 100,
             non_bench['benchmark_4hr_arrival'] * 100],
            color='b', lw=1, marker='o', alpha=0.6, markersize=4)

# Plot benchmark hospitals in red
mask = hospital_benchmark_rates['benchmark']
bench = hospital_benchmark_rates[mask]

for i, val in bench.iterrows():
    ax.plot([bench['actual_4hr_arrival'] * 100,
             bench['actual_4hr_arrival'] * 100],
            [bench['actual_4hr_arrival'] * 100,
             bench['benchmark_4hr_arrival'] * 100],
            color='r', lw=1, marker='o', alpha=0.6, markersize=4)


# Add mods 
ax.set_xlabel('Actual thrombolysis rate (%)')
ax.set_ylabel('Predicted benchmark thrombolysis rate (%)')
ax.set_xlim(0, 57)
ax.set_ylim(0, 57)

custom_lines = [Line2D([0], [0], color='r', alpha=0.6, lw=2),
                Line2D([0], [0], color='b', alpha = 0.6,lw=2)]

plt.legend(custom_lines, ['Benchmark team', 'Non-benchmark team'],
          loc='lower right')

plt.tight_layout()
plt.savefig('output/benchmark_thrombolysis.jpg', dpi=300)

plt.show()

### Calculated weighted average

Weight thrombolysis by admission numbers

base_4hr_arrival = ((hospital_benchmark_rates['count_4hr_arrival'] * 
                     hospital_benchmark_rates['actual_4hr_arrival']).sum() / 
                    hospital_benchmark_rates['count_4hr_arrival'].sum())

benchmark_4hr_arrival = ((hospital_benchmark_rates['count_4hr_arrival'] * 
                     hospital_benchmark_rates['benchmark_4hr_arrival']).sum() / 
                    hospital_benchmark_rates['count_4hr_arrival'].sum())

base_4hr_scan = ((hospital_benchmark_rates['count_4hr_scan'] * 
                     hospital_benchmark_rates['actual_4hr_scan']).sum() / 
                    hospital_benchmark_rates['count_4hr_scan'].sum())

benchmark_4hr_scan = ((hospital_benchmark_rates['count_4hr_scan'] * 
                     hospital_benchmark_rates['benchmark_4hr_scan']).sum() / 
                    hospital_benchmark_rates['count_4hr_scan'].sum())

print (f'Baseline thrombolysis onset-to-arrival of 4 hrs: {base_4hr_arrival*100:0.1f}')
print (f'Benchmark thrombolysis onset-to-arrival of 4 hrs: {benchmark_4hr_arrival*100:0.1f}')
ratio = benchmark_4hr_arrival / base_4hr_arrival
print (f'Benchmark thrombolysis ratio onset-to-arrival of 4 hrs: {ratio:0.3f}')
print ()
print (f'Baseline thrombolysis onset-to-scan of 4 hrs: {base_4hr_scan*100:0.1f}')
print (f'Benchmark thrombolysis onset-to-scan of 4 hrs: {benchmark_4hr_scan*100:0.1f}')
ratio = benchmark_4hr_scan / base_4hr_scan
print (f'Benchmark thrombolysis ratio onset-to-scan of 4 hrs: {ratio:0.3f}')

## Observations

* A 'benchmark' set of hospitals was created by identifying those 30 hospitals whith the highest predicted thrombolysis use in a set cohort of patients.

* The benchmark hospitals were not necessarily the top thrombolysing hospitals given their own patient populations.

* If decision to treat were made according to a majority vote of the benchmark set of hospitals then thrombolysis use (in those arriving within 4 hours of known stroke onset) would be expected to increase about 25%.