# Compare level of agreement in clinical decison making between hospitals using Random Forests models

## Aims:

* Quantify the level of agreement between different hospital models on what treatment patients should receive.

## Import libraries

# Turn warnings off to keep notebook tidy
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

## Load data

Create combined data set by combining cohort train/test

data = pd.concat([
    pd.read_csv('./../data/10k_training_test/cohort_10000_train.csv'),
    pd.read_csv('./../data/10k_training_test/cohort_10000_test.csv')],
    axis=0)

data = data.sample(frac=1.0, random_state=42)

hospitals = list(set(data['StrokeTeam'].values))

## Load previously saved models

with open ('./models/trained_hospital_models.pkl', 'rb') as f:
    hospital2model = pkl.load(f)

## Pass patients from each hospital through all other hospital models

# calculate results if not previously saved, else load previously saved results
calculate_results = True

# Calculate and save results if needed
if calculate_results:

    # create df for results
    results = pd.DataFrame(
        columns = np.concatenate((['True','Hospital'],hospitals)))
    
    results_prob = pd.DataFrame(
        columns = np.concatenate((['True','Hospital'],hospitals)))
    
    counter = 0

    # loop through patients from each hospital
    for hospital_test in hospitals:
        
        # Show progress
        counter += 1
        print (f'Calculating {counter} of {len(hospitals)}', end='\r')

        # df for each hospital's results
        hospital_results = pd.DataFrame(
            columns = np.concatenate((['True','Hospital'],hospitals)))
        
        hospital_results_prob = pd.DataFrame(
            columns = np.concatenate((['True','Hospital'],hospitals)))

        _, _, X_test, y_test = hospital2model[hospital_test]

        # put patients from hospital_test through all other hospital models
        for hospital_train in hospitals:

            if hospital_test==hospital_train:

                hospital_results['True'] = y_test
                hospital_results['Hospital'] = [hospital_test for v in y_test]
                hospital_results_prob['True'] = y_test
                hospital_results_prob['Hospital'] = [hospital_test for v in y_test]

            forest, threshold, _,_ = hospital2model[hospital_train]

            y_prob = forest.predict_proba(X_test)[:,1]

            y_pred = [1 if p >= threshold else 0 for p in y_prob]

            #add results from hospital_train to hospital df
            hospital_results[hospital_train] = y_pred
            hospital_results_prob[hospital_train] = y_prob
            

        #append hospital df to results df
        results = results.append(hospital_results, ignore_index=True)
        
        results_prob = results_prob.append(
            hospital_results_prob, ignore_index=True)

    #save results for later use
    results.to_csv('./predictions/hospital_results.csv', index=False)
    results_prob.to_csv('./predictions/hospital_results_prob.csv', index=False)

Load saved results

results = pd.read_csv(
    './predictions/hospital_results.csv')
results_prob = pd.read_csv(
    './predictions/hospital_results_prob.csv')

Add percent agreement for each patient

results['sum'] = results.sum(axis=1) - results['True']

results['percent'] = results['sum']*100/len(hospitals)

results['percent_agree'] = [max(p, 100-p) for p in results['percent']]

Show results (each patient is a row)

results.head()

## Analyse results

### Percentage of hospitals agreeing to give thrombolysis

results['percent']

fig = plt.figure(figsize=(5,5))

# Agreement on treatment 
ax1 = fig.add_subplot(111)
xvals = np.arange(0,101,1)

y1 = [sum([1 for p in results.percent.values if p>=x])\
         /len(results.percent.values) for x in xvals]

ax1.plot(xvals, y1)
ax1.set_xlabel('% Hospitals that choose to thrombolyse')
ax1.set_ylabel('Proportion of patients')
ax1.set_ylim(0, 1)
ax1.grid()


plt.tight_layout()
plt.savefig('./output/prop_agree_to_receive_thrombolysis.jpg', dpi=300)
plt.show()

### Proportion agreeing

Split patients by:

1. All patients - any agreement

2. Patients who did receive thrombolysis: percent of hospitals that would agree to give thrombolysis

3. Patients who did noit receive thrombolysis: percent of hospitals that would agree to not give thrombolysis

fig = plt.figure(figsize=(5,5))

# Agreement on treatment 
ax1 = fig.add_subplot(111)
xvals = np.arange(50,101,1)

y1 = [sum([1 for p in results.percent_agree.values if p>=x])\
         /len(results.percent_agree.values) for x in xvals]

ax1.plot(xvals, y1, label='Either treatment', linestyle='-')

y2 = [sum(1 for p in results[results['True']==1].percent_agree.values if p>=x)\
         /len(results[results['True']==1].percent_agree.values) for x in xvals]

ax1.plot(xvals, y2, label='Thrombolysed', linestyle='--')

y3 = [sum(1 for p in results[results['True']==0].percent_agree.values if p>=x)\
         /len(results[results['True']==0].percent_agree.values) for x in xvals]
      
ax1.plot(xvals, y3, label='Not thrombolysed', linestyle=':')
         
ax1.legend()

ax1.set_xlabel('% Hospitals that agree on given treatment')
ax1.set_ylabel('Proportion of patients')
ax1.set_ylim(0, 1)
ax1.grid()
      
plt.tight_layout()
plt.savefig('./output/agreement_x_hospital_single.jpg', dpi=300)
plt.show()

# Set percent of hospital to agree
xvals = [80, 90, 95]

# Find proportion of patients with agreement for either treat/no-treat decsion
agree_all = [sum([1 for p in results.percent_agree.values if p>=x])\
         /len(results.percent_agree.values) for x in xvals]

# Find proportion of thrombolysed patients with agreement
agree_thromb = [
    sum(1 for p in results[results['True']==1].percent_agree.values if p>=x)\
         /len(results[results['True']==1].percent_agree.values) for x in xvals]

# Find proportion of non-thrombolysed patients with agreement
agree_non_thromb = [
    sum(1 for p in results[results['True']==0].percent_agree.values if p>=x)\
         /len(results[results['True']==0].percent_agree.values) for x in xvals]

# Put in DataFrame
results_agree = pd.DataFrame()
results_agree['% Hospitals Agree'] = xvals
results_agree.set_index('% Hospitals Agree', inplace=True)
results_agree['Any'] = agree_all
results_agree['Throm'] = agree_thromb
results_agree['No Throm'] = agree_non_thromb
results_agree = results_agree.round(3)

results_agree

## Observations

* It is easier to find majority agreement on who not to thrombolyse than who to thrombolyse. For example:
    * 77.5% of all patients have a treatmnent decision that is agreed by 80% hospitals
    * Of those who were not given thrombolysis, 84.6% had agreement by 80% hospitals
    * Of those who were given thrombolysis, 60.4% had agreement by 80% hospitals 