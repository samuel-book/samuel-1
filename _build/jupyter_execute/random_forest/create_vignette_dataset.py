# Patient level decisions at own hospital, and consensus decisions at other other hospitals

This notebook creates two outputs:

1. Table of misclassified patients: patients with a predicted probability of being given thrombolysis (using a model of its own hospital) of <0.1 or >0.9, but where the hospital decided on the opposite. 

2. Vignette data: data for each patient including:

    * Patient feature data
    * Decision at own hospital
    * Predicted decision and probability at own hospital
    * Predicited consensus decision at the 30 benchmark hospitals
    * Consensus of whether 90% of other hospitals would thrombolyse
    * Consensus of whether 80% of other hospitals would thrombolyse
    * Average probability of receiving thrombolysis at other hospitals

## Import libraries 

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Load arrivals within 4 hours (load train and test cohort sets and combine)
train = pd.read_csv('../data/10k_training_test/cohort_10000_train.csv')
test = pd.read_csv('../data/10k_training_test/cohort_10000_test.csv')
data = pd.concat([train, test], axis=0)

## Fit models and find confident misclassifications

Models are first fitted using k-fold cross-validation. 'Confident misclassifications' are patients with a predicted probability of being given thrombolysis (using a model of its own hospital) of <0.1 or >0.9, but where the hospital decided on the opposite. These are taken from test sets only.

The k-fold splits are also used to predict and record the probability of being given thrombolysis using the test set patients only (which is all patients across the 5 splits). Whether a patient is given thrombolysis or not is predicted from the threshold that matches thrombolysis rate in the test set to the observed rate for that hospital.

After the k-fold validation prediction, a model is fitted to all data for a hospital. This is used for counter-factual predictions later (hat treatment patients from other hospitals would be predicted to receive at that hospital).

hospitals = list(set(data['StrokeTeam'].values))

def find_threshold(probabilities, true_rate):
    
    """
    Find classification threshold to calibrate model
    """
    
    index = (1-true_rate)*len(probabilities)
    
    threshold = sorted(probabilities)[int(index)]
    
    return threshold

%%time

hospital2model={}

misclassifications = pd.DataFrame(columns = data.columns.values)

for hospital in hospitals:
    
    y_probs, test_idx = [],[]
    
    patients = data.loc[data['StrokeTeam'] == hospital]
    
    y = patients['S2Thrombolysis']
    X = patients.drop(['StrokeTeam','S2Thrombolysis'], axis=1)
    
    skf = StratifiedKFold(n_splits = 5)
    skf.get_n_splits(X, y)
    
    for train_index, test_index in skf.split(X, y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                        class_weight='balanced', random_state=0)

        forest.fit(X_train, y_train)

        y_prob = forest.predict_proba(X_test)[:,1]
        
        y_probs.extend(y_prob)
        
        test_idx.extend(test_index)
        
        for i,p in enumerate(y_prob):
        
            if p>=0.9 and y_test.values[i]==0:
                
                misclassifications = misclassifications.append(
                    patients.iloc[test_index[i]], ignore_index=True)

            if p<=0.1 and y_test.values[i]==1:

                misclassifications = misclassifications.append(
                    patients.iloc[test_index[i]], ignore_index=True)
                
    true_rate = sum(y)/len(y) 

    threshold= find_threshold(y_probs, true_rate)
    
    y_preds = [1 if p >= threshold else 0 for p in y_probs]

    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                        class_weight='balanced', random_state=0)

    forest.fit(X.values, y.values)
    
    hospital2model[hospital]= [forest, threshold, y_probs, y_preds, test_idx]

misclassifications.head()

misclassifications.to_csv('./output/misclassifications.csv')

## Combine consensus, benchmark and predicted outcomes for all patients

Each patient has the predicted probability of receiving thrombolysis at all hospitals other than the one they actually attended. 

* *Benchmark*: The consensus decision of the 30 benchmark hospitals (thrombolysis predicted to be given if at least 15 of the 30 benchmark hospitals have predicted thrombolysis use for that hospital).

* *Majority_90*: True if at least 90% of other hospitals are predicted to give thrombolysis.

* *Majorit_ 80*: True if at least 80% of other hospitals are predicted to give thrombolysis.

* *Predicted*: Whether the patient's own hospital is predicted to give thrombolysis (as recoded in the k-fold predictions above). Depends on both probability of receiving thrombolysis and the threshold of prediction in that k-fold split that would give the same net thrombolysis use as the hospital overall.

* *Probability*: Whether the patient's own hospital is predicted to give thrombolysis (as recoded in the k-fold predictions above).

* *AvgProbability*: The average probability of other hospitals giving thrombolysis to a patient.

benchmark_hospitals = np.load(
    './models/benchmark_hospitals.npy', allow_pickle=True)

%%time

#create df for results
results = pd.DataFrame(columns = np.concatenate((data.columns.values,\
                        ['Benchmark','Majority_90','Majority_80','Predicted'])))

#loop through patients from each hospital
for hospital in hospitals:
    
    # df for each hospital's results
    hospital_results = pd.DataFrame(columns = hospitals)
    hospital_results_prob = pd.DataFrame(columns = hospitals)
    
    patients = data.loc[data['StrokeTeam'] == hospital]
    
    _, _, _, _, patient_order = hospital2model[hospital]
    
    patients = patients.iloc[patient_order]
    
    y = patients['S2Thrombolysis']
    X = patients.drop(['StrokeTeam','S2Thrombolysis'], axis=1)
    
    #put patients from hospital through all other hospital models
    for hospital_train in hospitals:
        
        forest, threshold, _, _, _ = hospital2model[hospital_train]
        
        if hospital==hospital_train:
            
            continue
        
        y_prob = forest.predict_proba(X)[:,1]
   
        y_pred = [1 if p >= threshold else 0 for p in y_prob]
        
        #add results from hospital_train to hospital df
        hospital_results[hospital_train] = y_pred
        hospital_results_prob[hospital_train] = y_prob
        
    patients_copy = patients.copy()
    
    #add outcome from benchmark hospitals
    
    benchmark_results = hospital_results[benchmark_hospitals].copy()
    
    benchmark_results['sum'] = benchmark_results.sum(axis=1)
    
    benchmark_results['percent'] = benchmark_results['sum']*100/len(benchmark_hospitals)
    
    benchmark_outcome = [1 if p>=50 else 0 for p in benchmark_results['percent']]
    
    patients_copy['Benchmark'] = benchmark_outcome
        
    #majority outcomes
        
    hospital_results['sum'] = hospital_results.sum(axis=1)

    hospital_results['percent'] = hospital_results['sum']*100/len(hospitals)

    #90% majority

    majority_90 = [1 if p>=90 else 0 for p in hospital_results['percent']]
    
    patients_copy['Majority_90'] = majority_90
    
    #80% majority
    
    majority_80 = [1 if p>=80 else 0 for p in hospital_results['percent']]
    
    patients_copy['Majority_80'] = majority_80
    
    #predicted outcomes
    
    _, _, probability, predicted, _ = hospital2model[hospital]
    
    patients_copy['Predicted'] = predicted
    patients_copy['Probability'] = probability
    
    #average probability
    
    hospital_results_prob[hospital] = probability
    patients_copy['AvgProbability'] = hospital_results_prob.mean(axis=1).round(2).values
    
    #append to results df
    
    results = results.append(patients_copy, ignore_index=True)

results

results.to_csv('./output/vignette_data.csv')