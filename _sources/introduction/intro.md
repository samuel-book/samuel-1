# Introduction to SAMueL-1

*"Your decision to treat or not treat… That’s the difficult part. That’s the grey area where everyone does a different thing."* Stroke Consultant during interviews for SAMueL.

This Jupyter Book contains code for the SAMueL-1 project: Use of simulation and machine learning to identify key levers for maximising the disability benefit of intravenous thrombolysis in acute stroke pathways.

NIHR HS&DR Reference Number: 17/99/89

Note: This repository and book does not contain the original data, as that is patient-level and considered sensitive. We are working to make synthetic data available.

Data used is for units which have at least 300 emergency stroke admissions with at least 10 pateints receiving thrombolysis in three years (2016-2018).

## The stroke thrombolysis pathway

![](./../images/pathway.png)

For modelling we use key stages in the thrombolysis pathway. These are:

1. A patient has a stroke. They call for an ambulance (or they may convey themselves to hospital). In this study we do not generally consider the 5% of strokes that occur in-hospital, as the pathway may often be different.

1. Once a patient arrives at hospital, staff gather information on the patient and attempt to determine the time of onset (and make a note whether they consider this time precise or a best estimate).

1. The patient has a head scan to determine whether the stroke is caused by a clot (about 85% of all strokes) or a bleed (the remaining 15%). Clot-busting drugs, or clot-removal surgery can onlt be used on strokes caused by a clot.

1. Following the scan the clinicians will decide whether to treat with clot-busting drugs ('thrombolysis'). These are best given as soon as possible after a stroke and usually not past four and a half hours after the stroke. For this reason in much of our work we restrict modelling the thrombolysis pathway to those patients arriving at hospital within four hours of stroke onset.

## Aims of project (plain English)

### Aim

This study aims to develop a tool for doctors to help them review and improve the speed and use of a clot busting drug when someone has had a stroke. 

### Background

Stroke is a leading cause of death and disability, with over 85,000 people hospitalised in the UK each year. One way of treating stroke and prevent disability is to give the patient medication that breaks down blood clots. This is called thrombolysis. 

Thrombolysis is not suitable for all patients, and can be risky. For thrombolysis to be useful it needs to be given as soon after the stroke as possible. The use of thrombolysis varies hugely, even for patients with similar treatment pathways and with similar characteristics. Some hospitals rarely use it, some use it in a quarter of stroke patients. The speed of giving thrombolysis also varies. Some hospitals take an average of 90 minutes, others less than 40 minutes, to administer the drug.

We will find out why there is so much variability in the use of thrombolysis. This will help hospitals understand what they can do to optimise their use of the drug. Based on the decisions made by highly qualified stroke experts we will build a tool for assisting doctors on the use of thrombolysis. This will be particularly useful to hospitals which do not have enough funds to employ a team of stroke physicians.

### Study design

The research team will use a state-of-the-art computer modelling technique to better understand what is causing the variation in care across the UK. This technique is called pathway modelling. With this approach we replicate, in a computer model, the flow of patients through the first few hours of stroke care, mimicking the same processes and timings that the stroke unit currently provides.  This allows us to look at the effect of changing key aspects of the patient flows in a controlled, modelled environment, without affecting real patients. 

A second technique called ‘machine learning’ enables us to teach a computer the likely decision made in any hospital given any particular patient. Both approaches allow us to ask 'what if?' questions, such as 'what if a hospital improved diagnosis of patients by asking more questions, but thereby delaying patients’ time to receive a scan?' With machine learning we can ask ‘what if the decisions at all hospitals were similar to hospitals that are considered centres of excellence for stroke care?’ By asking these types of questions we can identify changes at each hospital which would most benefit patients. Both methods have been piloted across seven hospitals. We would now like to test and refine these methods across all stroke units in England. 

A researcher will interview doctors to understand their attitudes to thrombolysis, and how the results from the modelling work can best be presented to them in a way that will influence more consistent stroke care across the UK.

We will work closely with the national stroke audit team to make sure the models can be used as part of their established work. 

## Creative Commons

This work is shared under CC BY 4.0.

You are free to:

* Share - copy and redistribute the material in any medium or format

* Adapt - remix, transform, and build upon the material for any purpose, even commercially.

You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

License info: https://creativecommons.org/licenses/by/4.0/
