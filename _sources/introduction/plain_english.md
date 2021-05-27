# Plain English Summary

**BACKGROUND**

Stroke is a common cause of adult disability. Expert opinion is that about one in five patients should receive clot-busting drugs to break up the blood clot that is causing their stroke. This is called *thrombolysis*. At the moment only about one in nine patients actually receive this treatment in the UK. There is a lot of variation between hospitals, which means that the same patient might receive different treatment in different hospitals. 

Clot-busting drugs are not suitable for everyone. There is a small risk of a bleed in the brain. Doctors must feel confident in their use, and lack of confidence may explain some of the variation in use. In previous work we developed the basic methods for understanding what are the main causes of variation between hospitals: How much difference is due to processes (like how quickly a patient is scanned, an essential step), how much is due to differences in patient populations, and how much difference is due to different decision-making by doctors. This has enabled us to model the ideal number of patients who should be treated in each hospital in the UK.

**WHAT WE DID**

Our work broke down into three areas.

1. We built *machine learning* models that learned which patients would be given thrombolysis at each hospital. *Machine learning* works on the principle of asking *"What happened to similar patients in this hospital before?"* 

2. We built a simulation of stroke pathways for all hospitals. This simulation would pass individual patients through a pathway with each patient having their own unique speed of movement through the pathway, but with speeds all being typical of a particular hospital. Each hospital also has patients that are typical of that hospital (replicating, for example, the number of patients who arrive in hospital in time for clot-busting treatment). With this simulation we could ask questions like *"What would happen if we could speed up the pathway at this hospital?*.

3. We interviewed doctors about what we were doing. We asked for their feedback - what did  they like and what did they not like? What is most useful? What are we missing?


**WHAT WE FOUND**

Some of our most important findings were:

* Our machine learning models were correct eight to nine times out of ten. They also report how sure they are about a decision, and the more sure they are the more often they are right.

* Using our machine learning models we could ask the question *"What treatment would this patient have likely received at different hospitals?"*

* Different hospitals would likely make different decisions on patients. Hospitals can more easily agree on who definitely *should not be* treated with clot-busting drugs than who *should be*.

* There is large range in *willingness to treat*. Given the same set of patients, those most willing to treat would treat four to five times more patients than those least willing to treat.

* If all hospitals made decisions like those of the top 30 *most willing to treat* hospitals, there would be 13 people treated for every 10 treated now. And that would mean there would be 13 disability-free outcomes due to treatment for every 10 now.

* We can show individual hospitals examples of patients that they appear to treat differently from the top 30 *most willing to treat* hospitals. We hope this will help open up discussions on why different hospitals select different patients for treatment.

* We can highlight patients who were not treated as we would expect them to be treated in the hospital that they attended. This might be due to us not having all the important information, but it might also help hospitals see the patients where the pathway did not perform well, and help them look to see how they could make it better in the future.

* Determining the time a stroke started is critical if clot-busting drugs are to be given. Some hospitals do that for many more patients than others. Sometimes this will be because a hospital has patients where this is very hard to find, but other times it may because some hospitals have worked out how to do this best, along with their local ambulance colleagues. If all hospitals managed to do as well as a typical 'good' hospital (about a quarter the way down the 'league table' of how many patients they determine the stroke onset time for), there would be 11-12 people treated for every 10 treated now.

* Before a patient can be treated they must have a brain scan, and then the treatment must be prepared and delivered. Hospitals manage to do this at different speeds. If all hospitals managed to do both of those tasks in 30 minutes (which some hospitals have shown is possible) then there would be 12-13 people treated for every 10 treated now. But as well as treating more patients, all patients would be treated more quickly, getting more benefit from the drug, so there would be 15 disability-free outcomes due to treatment for every 10 now.

* If we combined all these changes across all hospitals then we would expect there would be 16 people treated for every 10 treated now. That is instead of treating about one in nine patients, as now, we would treat one in five to six. More importantly, there would be 19 disability-free outcomes due to treatment for every 10 now.

* After we test all these changes at each hospital in our models we find there will still be quite a lot of variation in how many are treated at each hospital. This is because hospitals have different populations of patients. So rather than having a single target for treatment for all hospitals, it may be better to have a realistic target for each hospital, which takes their patient mix into account.

* For each hospital we can predict which change would make most difference - is it speed, or willingness to treat, or determining stroke onset times? We can help hospitals identify what is best for them to work on improving, and that may be different to the hospital down the road.

* Feedback from doctors throughout the project was really positive and useful. They helped us to work out what to show. They also had questions for us on how we could take this forward. They wanted to know how doctors would know about this. They wanted to know how we would communicate with managers and the people who pay for the hospitals if they needed investment to improve the service. Importantly, they wanted our machine learning models to learn to predict likely outcome, as well as likely treatment decision.

We are working with the 'National Stroke Audit' now on making our models part of their routine reporting back to hospitals. And we are, of course, working on what we shall do next to develop this work further!

