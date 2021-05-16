# Extract hospital performance for pathway model

## Aims

* Extract and save hospital performance for pathway simulation model
* Create breakdowns by weekend/weekday/day/night

## Import libraries

import numpy as np
import pandas as pd

## Load data

* Load data
* Restrict data to fields necessary for pathway extraction
* Remove in-hospital admissions

# Load data
data_loaded = pd.read_csv(
    '../data/2019-11-04-HQIP303-Exeter_MA.csv', low_memory=False)

# Number of years data covers
data_years = 3.0

# Restrict fields
used_fields = [
    'StrokeTeam',
    'MoreEqual80y',
    'S1Gender',
    'S1OnsetInHospital',
    'S1OnsetToArrival_min',
    'S1AdmissionHour',
    'S1AdmissionDay',
    'S1OnsetTimeType',
    'S2BrainImagingTime_min',
    'S2StrokeType',
    'S2Thrombolysis',
    'S2ThrombolysisTime_min']

data_loaded = data_loaded[used_fields]

# Remove in hospital admissions
mask = data_loaded['S1OnsetInHospital'] == 'No'
data_loaded = data_loaded[mask]

## Extract hospital performance

def analyse_by_team(input_data):
    
    # Copy data
    data = input_data.copy()
    
    # Set up results lists
    stroke_team = []
    admissions = []
    age_80_plus = []
    onset_known = []
    known_arrival_within_4hrs = []
    onset_arrival_mins_mu = []
    onset_arrival_mins_sigma = []
    scan_within_4_hrs = []
    arrival_scan_arrival_mins_mu = []
    arrival_scan_arrival_mins_sigma = []
    onset_scan_4_hrs = []
    scan_needle_mins_mu = []
    scan_needle_mins_sigma = []
    thrombolysis_rate = []
    eligible = []
    
    # Split data by stroke team
    groups = data.groupby('StrokeTeam') # creates a new object of groups of data
    group_count = 0
    for index, group_df in groups: # each group has an index + dataframe of data
        group_count += 1

        # Record stroke team
        stroke_team.append(index)

        # Record admission numbers
        admissions.append(group_df.shape[0])

        # Get thrombolysis rate
        thrombolysed = group_df['S2Thrombolysis'] == 'Yes'
        thrombolysis_rate.append(thrombolysed.mean())

        # Record onset known proportion and remove rest
        f = lambda x: x in ['Precise', 'Best estimate']
        mask = group_df['S1OnsetTimeType'].apply(f)
        onset_known.append(mask.mean())
        group_df = group_df[mask]

        # Record onset <4 hours and remove rest
        mask = group_df['S1OnsetToArrival_min'] <= 240
        known_arrival_within_4hrs.append(mask.mean())
        group_df = group_df[mask]

        # Calc proportion 80+ (of those arriving within 4 hours)
        age_filter = group_df['MoreEqual80y'] == 'Yes'
        age_80_plus.append(age_filter.mean())

        # Log mean/sd of onset to arrival
        ln_onset_to_arrival = np.log(group_df['S1OnsetToArrival_min'])
        onset_arrival_mins_mu.append(ln_onset_to_arrival.mean())
        onset_arrival_mins_sigma.append(ln_onset_to_arrival.std())

        # Record scan within 4 hours of arrival (and remove the rest)
        mask = group_df['S2BrainImagingTime_min'] <= 240
        scan_within_4_hrs.append(mask.mean())
        group_df = group_df[mask]
        
        # Log mean/sd of arrival to scan
        ln_arrival_to_scan = np.log(group_df['S2BrainImagingTime_min'])
        arrival_scan_arrival_mins_mu.append(ln_arrival_to_scan.mean())
        arrival_scan_arrival_mins_sigma.append(ln_arrival_to_scan.std())
        
        # Record onset to scan in 4 hours and remove rest
        mask = (group_df['S1OnsetToArrival_min'] + 
                group_df['S2BrainImagingTime_min']) <= 240
        onset_scan_4_hrs.append(mask.mean())
        group_df = group_df[mask]

        # Thrombolysis given (to remaining patients)
        thrombolysed = group_df['S2Thrombolysis'] == 'Yes'
        eligible.append(thrombolysed.mean())

        # Scan to need (Replace any zero scan to needle times with 1)
        mask = group_df['S2ThrombolysisTime_min'] > 0
        thrombolysed = group_df[mask]
        scan_to_needle = (thrombolysed['S2ThrombolysisTime_min'] - 
                          thrombolysed['S2BrainImagingTime_min'])
        mask = scan_to_needle == 0
        scan_to_needle[mask] = 1
        ln_scan_to_needle = np.log(scan_to_needle)
        scan_needle_mins_mu.append(ln_scan_to_needle.mean())
        scan_needle_mins_sigma.append(ln_scan_to_needle.std())
        
    df = pd.DataFrame()
    df['stroke_team'] = stroke_team
    df['thrombolysis_rate'] = thrombolysis_rate
    df['admissions'] = admissions
    df['admissions'] = df['admissions'] /data_years
    df['80_plus'] = age_80_plus
    df['onset_known'] = onset_known
    df['known_arrival_within_4hrs'] = known_arrival_within_4hrs
    df['onset_arrival_mins_mu'] = onset_arrival_mins_mu
    df['onset_arrival_mins_sigma'] = onset_arrival_mins_sigma
    df['scan_within_4_hrs'] = scan_within_4_hrs
    df['arrival_scan_arrival_mins_mu'] = arrival_scan_arrival_mins_mu
    df['arrival_scan_arrival_mins_sigma'] = arrival_scan_arrival_mins_sigma
    df['onset_scan_4_hrs'] = onset_scan_4_hrs
    df['eligable'] = eligible
    df['scan_needle_mins_mu'] = scan_needle_mins_mu
    df['scan_needle_mins_sigma'] = scan_needle_mins_sigma
    
    return df
    

df_all = analyse_by_team(data_loaded)

# Limit to hosp with > 100 admissions/year and >10 thrombolysis in total
admissions = df_all['admissions']
thrombolysed = admissions * df_all['thrombolysis_rate']
mask = (admissions >= 100) & (thrombolysed >= 3.3333)
df_all = df_all[mask]

# Save
df_all.to_csv('hosp_performance_output/hospital_performance.csv', index=False)

# Show data for five hopsitals
df_all.head().T

### Limit full data to units with at least 300 admissions

units_with_300_admissions = list(set(df_all['stroke_team']))
mask = data_loaded['StrokeTeam'].isin(units_with_300_admissions)
data_restricted = data_loaded[mask]

### Produce results for day/night and weekday/weekend

day_time_values = ['09:00 to 11:59', '12:00 to 14:59', '15:00 to 17:59']
values = data_restricted['S1AdmissionHour'].isin(day_time_values)
data_restricted = data_restricted.assign(day_time=values)    

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
values = data_restricted['S1AdmissionDay'].isin(weekdays)
data_restricted = data_restricted.assign(mon_fri=values)

Weekday

mask = data_restricted['mon_fri']
df = data_restricted[mask]
df = analyse_by_team(df)
df.to_csv(
    'hosp_performance_output/hospital_performance_weekday.csv', index=False)

Weekday day

mask = data_restricted['day_time'] & data_restricted['mon_fri']
df = data_restricted[mask]
df = analyse_by_team(df)
df.to_csv(
    'hosp_performance_output/hospital_performance_weekday_day.csv', index=False)

Weekday night

mask = data_restricted['day_time'] == False & data_restricted['mon_fri']
df = data_restricted[mask]
df = analyse_by_team(df)
df.to_csv(
    'hosp_performance_output/hospital_performance_weekday_night.csv', index=False)

Weekend

mask = data_restricted['mon_fri'] == False
df = data_restricted[mask]
df = analyse_by_team(df)
df.to_csv(
    'hosp_performance_output/hospital_performance_weekend.csv', index=False)

Weekend day

mask = data_restricted['day_time'] & data_restricted['mon_fri'] == False
df = data_restricted[mask]
df = analyse_by_team(df)
df.to_csv(
    'hosp_performance_output/hospital_performance_weekend_day.csv', index=False)

Weekend night

mask = (
    data_restricted['day_time'] == False) & (data_restricted['mon_fri'] == False)
df = data_restricted[mask]
df = analyse_by_team(df)
df.to_csv(
    'hosp_performance_output/hospital_performance_weekend_night.csv', index=False)