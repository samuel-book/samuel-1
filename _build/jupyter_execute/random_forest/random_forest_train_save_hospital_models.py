# Train and save hospital-level models

## Aims

* Train models for each hospital.

Note: Models are trained on all data for maximum accuracy. These models are used for counter-factual experiments (what if patient went to another hospital).

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

## Functions

### Find model probability threshold to match predicted and actual thrombolysis use

def find_threshold(probabilities, true_rate):
    
    """
    Find classification threshold to calibrate model
    """
    
    index = (1-true_rate)*len(probabilities)
    
    threshold = sorted(probabilities)[int(index)]
    
    return threshold

## Train hospital-level models

Get threshold of classification by using k-fold splits: train hospital level model on k-fold splits, get probabilities of classification for test sets, and use these together to get probability threshold that gives the same net thrombolysis use as the observed data. Then refit on all data for a given hospital, and return that fit and the hospital model.

def train_hospital(data, hospital):

    # Get data for given hospital
    patients = data.loc[data['StrokeTeam'] == hospital]
    
    # Get X and y
    y = patients['S2Thrombolysis']
    X = patients.drop(['StrokeTeam','S2Thrombolysis'], axis=1)
    
    # Split 5-fold
    skf = StratifiedKFold(n_splits = 5, random_state=42)
    skf.get_n_splits(X, y)
    
    # Set up list for predicted test probabilities
    y_probs=[]
    
    # Train on 5-fold splits and get probabilities
    for train_index, test_index in skf.split(X, y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                        class_weight='balanced', random_state=0)


        forest.fit(X_train, y_train)

        y_prob = forest.predict_proba(X_test)[:,1]
        
        y_probs.extend(y_prob)
        
    # Get observed thrombolysis use rate
    true_rate = sum(y)/len(y) 

    # Find threshold of probability to match observed thrombolysis use
    threshold = find_threshold(y_probs, true_rate)

    # Refit on all data for hospital
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                        class_weight='balanced', random_state=0)

    forest.fit(X.values, y.values)

    # Return fit, threshold, and X/y for given hospitalk
    return [forest, threshold, X.values, y.values]

## Train model on each hospital (and save models)

Note: train_save is a variable used to avoid retraining models when publishing Jupyter Book. If new saved models are needed change this variable to True

# Cahnge this to True to refit and save models
train_save = False

if train_save:

    # Get hospitals
    hospitals = list(set(data['StrokeTeam'].values))

    # Set up dictionsary for models
    hospital2model = {}

    # Train models
    for hospital in hospitals:
        hospital2model[hospital] = train_hospital(data, hospital)
    
    # Save model, threshold and data
    with open ('./models/trained_hospital_models.pkl', 'wb') as f:
        pkl.dump(hospital2model, f)  