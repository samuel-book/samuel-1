# Introduction to SAMueL (Stroke Audit Machine Learning)

**Use of simulation and machine learning to identify key levers for maximising the disability benefit of intravenous thrombolysis in acute stroke pathways**

```{epigraph}
"Your decision to treat or not treat … That’s the difficult part. That’s the grey area where everyone does a different thing."

-- Stroke Consultant during interviews for SAMueL
```

## Background

Stroke is a common cause of adult disability. Expert opinion is that about one in five patients should receive clot-busting drugs to break up the blood clot that is causing their stroke, and this is the target use set in the NHS long term plan. This clot-busting treatment is called thrombolysis. At the moment only about one in nine patients actually receive this treatment in the UK. There is a lot of variation between hospitals, which means that the same patient might receive different treatment in different hospitals.

Clot-busting drugs are not suitable for everyone. There is a small risk of a bleed in the brain. Doctors must feel confident in their use, and lack of confidence may explain some of the variation in use. In our work we have developed the methods for understanding what are the main causes of variation between hospitals: How much difference is due to processes (like how quickly a patient is scanned, an essential step), how much is due to differences in patient populations, and how much difference is due to different decision-making by doctors. This has enabled us to model the a realistic number of patients who could or should be treated in each hospital in the UK. We predict that the number of people for whom clot-busting drugs would prevent disability after stroke could be nearly doubled.

{numref}`Figure {number} <fig_pathway_net>` shows the net effect of changes at all hospitals:

:::{figure-md} fig_stickmen
<img src="./../images/stick_men.png" width="600px">

A summary of the peceived problem. Clinical expert opinion is that two in every ten emergency stroke patients could be treated be thrombolysis, but the reality is that only about one in ten are treated with thrombolysis. 
:::

### Overall aims:

The overall aims of the project are

* Model thrombolysis decision-making using machine learning, so that we may ask the question *'What treatment would my patient receive at other hospitals?'*

* Model the stroke pathway, using clinical pathway simulation, so that we may ask the question *"What would happen to a hospital's thrombolysis use of, and benefit from, thrombolysis by changing key aspects of the pathway?"*, especially focussing on:

    * Pathway speed
    
    * Determining stroke onset times
    
    * Making decisions according to the majority vote of decisions that would be expected at a *benchmark* set of hospitals
    
This Jupyter Book contains methods (code) and results for the SAMueL-1 project: *Use of simulation and machine learning to identify key levers for maximising the disability benefit of intravenous thrombolysis in acute stroke pathways.* NIHR HS&DR Reference Number: 17/99/89
