# Pathway patterns throughout the week

## Aims

Show key pathway statistics broken down by day of week

## Import libraries and data

Data has been resticted to stroke teams with at least 300 admissions, with at least 10 patients receiving thrombolysis, over three years.

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Display entire dataframes
pd.set_option("display.max_rows", 999, "display.max_columns", 150)

# import data
raw_data = pd.read_csv(
    './../data/2019-11-04-HQIP303-Exeter_MA.csv', low_memory=False)

headings = list(raw_data)
print (headings)

## Restrict original data to hospitals with at least 300 admissions + 10 thrombolysis patients

keep = []

groups = raw_data.groupby('StrokeTeam') # creates a new object of groups of data

for index, group_df in groups: # each group has an index and a dataframe of data
    
    # Skip if total admissions less than 300 or total thrombolysis < 10
    admissions = group_df.shape[0]
    thrombolysis_received = group_df['S2Thrombolysis'] == 'Yes' 
    if (admissions < 300) or (thrombolysis_received.sum() < 10):
        continue
    
    else: 
        keep.append(group_df)

# Concatenate output
data = pd.DataFrame()
data = pd.concat(keep)

Remove in-hospital onset

mask = data['S1OnsetInHospital'] == 'No'
data = data[mask]

## Group by day of week

# Work on copy of data
data_time = data.copy()

# Set up results DataFrame
day_summary = pd.DataFrame()

# Count arrivals
day_summary['Arrivals'] = \
    data_time.groupby('S1AdmissionDay').count()['StrokeTeam']

# Get thrombolysis rate
thrombolysed = data_time['S2Thrombolysis'] == 'Yes'
data_time['thrombolysed'] = thrombolysed
day_summary['thrombolyse_all'] = \
    data_time.groupby('S1AdmissionDay').mean()['thrombolysed']

# Get proportion of strokes with known onset
onset_known = (data_time['S1OnsetTimeType'] == 'Best estimate') | \
    (data_time['S1OnsetTimeType'] == 'Precise')
data_time['onset_known'] = onset_known
day_summary['onset_known'] = \
    data_time.groupby('S1AdmissionDay').mean()['onset_known']

# Get proportion over 80
data_time['age_80_plus'] = data_time['MoreEqual80y'] == 'Yes'
day_summary['age_80_plus'] = \
    data_time.groupby('S1AdmissionDay').mean()['age_80_plus']

# Get Rankin score
day_summary['rankin_all'] = \
    data_time.groupby('S1AdmissionDay').mean()['S2RankinBeforeStroke']

# Get NIHSS
day_summary['nihss_all'] = \
    data_time.groupby('S1AdmissionDay').mean()['S2NihssArrival']

# Get onset to arrival <4hrs and then restrict data
data_time['4hr_arrival'] = data_time['S1OnsetToArrival_min'] <= 240
day_summary['4hr_arrival'] = \
    data_time.groupby('S1AdmissionDay').mean()['4hr_arrival']
mask = data_time['4hr_arrival']
data_time = data_time[mask]

# Get Rankin score of arrivals within 4hrs onset
day_summary['rankin_4hr'] = \
    data_time.groupby('S1AdmissionDay').mean()['S2RankinBeforeStroke']

# Get NIHSS of arrivals within 4hrs onset
day_summary['nihss_4hr'] = \
    data_time.groupby('S1AdmissionDay').mean()['S2NihssArrival']

# Get onset to arrival (of those arriving within 4 hours)
day_summary['onset_arrival'] = \
    data_time.groupby('S1AdmissionDay').mean()['S1OnsetToArrival_min']

# Get scan in four hours (and remove rest)
data_time['4hr_scan'] = data_time['S2BrainImagingTime_min'] <= 240
day_summary['scan_4hrs'] = \
    data_time.groupby('S1AdmissionDay').mean()['4hr_scan']
mask = data_time['4hr_scan']
data_time = data_time[mask]

# Get arrival to scan (of those arriving within 4 hours and scanned in 4hrs)
day_summary['arrival_scan'] = \
    data_time.groupby('S1AdmissionDay').mean()['S2BrainImagingTime_min']

# Filter down to acanned within 4 hrs onset
onset_to_scan = (
    data_time['S1OnsetToArrival_min'] + data_time['S2BrainImagingTime_min'])
data_time['onset_to_scan'] = onset_to_scan
mask = data_time['onset_to_scan'] <= 240
data_time = data_time[mask]

# Get thrombolysis given and remove rest
day_summary['thrombolyse_4hr'] = \
    data_time.groupby('S1AdmissionDay').mean()['thrombolysed']
mask = data_time['thrombolysed']

# Get scan to needle
scan_to_needle = (
    data_time['S2ThrombolysisTime_min'] - data_time['S2BrainImagingTime_min'])
data_time['scan_needle'] = scan_to_needle
day_summary['scan_to_needle'] = \
    data_time.groupby('S1AdmissionDay').mean()['scan_needle']

# Get arrival to needle
day_summary['arrival_to_needle'] = \
    data_time.groupby('S1AdmissionDay').mean()['S2ThrombolysisTime_min']

# Get onset to needle 
onset_to_needle = (
    data_time['S1OnsetToArrival_min'] + data_time['S2ThrombolysisTime_min'])
data_time['onset_to_needle'] = onset_to_needle
day_summary['onset_to_needle'] = \
    data_time.groupby('S1AdmissionDay').mean()['onset_to_needle']

# Sort by day
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
            'Sunday']


day_summary = day_summary.loc[day_order]
day_summary['Weekday'] = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

## Show summary table by day of week

day_summary.T

## Show summary charts of key metrics by day of week

# Set up figure
fig = plt.figure(figsize=(12,15))

# Subplot 1: Arrivals
ax1 = fig.add_subplot(4,4,1)
x = day_summary['Weekday']
y = day_summary['Arrivals'] / day_summary['Arrivals'].sum() * 7
ax1.plot(x,y)
# Add line at 1
y1 = np.repeat(1,7)
ax1.plot(x,y1, color='0.5', linestyle=':')
ax1.set_ylim(ymin=0) # must be after plot method
ax1.set_xlabel('Day of week')
ax1.set_ylabel('Normlalised arrivals')
ax1.set_title('Arrivals\n(normalised to average)')

# Subplot 2: Thrombolysis
ax2 = fig.add_subplot(4,4,2)
x = day_summary['Weekday']
y = day_summary['thrombolyse_all'] * 100
ax2.plot(x,y)
ax2.set_xlabel('Day of week')
ax2.set_ylabel('Thrombolysis (%)')
ax2.set_title('Thrombolysis use\n(all arrivals)')

# Subplot 3: Known onset
ax3 = fig.add_subplot(4,4,3)
x = day_summary['Weekday']
y = day_summary['onset_known'] * 100
ax3.plot(x,y)
ax3.set_xlabel('Day of week')
ax3.set_ylabel('Onset known (%)')
ax3.set_title('Proportion with known onset')

# Subplot 4: age_80_plus 
ax4 = fig.add_subplot(4,4,4)
x = day_summary['Weekday']
y = day_summary['age_80_plus'] * 100
ax4.plot(x,y)
ax4.set_xlabel('Day of week')
ax4.set_ylabel('Aged 80+')
ax4.set_title('Proportion aged 80+')

# Subplot 5: Rankin (all arrivals)
ax5 = fig.add_subplot(4,4,5)
x = day_summary['Weekday']
y = day_summary['rankin_all'] 
ax5.plot(x,y)
ax5.set_xlabel('Day of week')
ax5.set_ylabel('Rankin (mean)')
ax5.set_title('Mean pre-stroke\nmodified Rankin\n(all arrivals)')

# Subplot 6: NIHSS (all arrivals)
ax6 = fig.add_subplot(4,4,6)
x = day_summary['Weekday']
y = day_summary['nihss_all'] 
ax6.plot(x,y)
ax6.set_xlabel('Day of week')
ax6.set_ylabel('NIHSS (mean)')
ax6.set_title('Mean NIH Stroke Scale\n(all arrivals)')

# Subplot 7: 4hr_arrival
ax7 = fig.add_subplot(4,4,7)
x = day_summary['Weekday']
y = day_summary['4hr_arrival'] * 100
ax7.plot(x,y)
ax7.set_xlabel('Day of week')
ax7.set_ylabel('Arrive within 4hrs of onset (%)')
ax7.set_title('Proportion arriving within\n4hrs of known onset')

# Subplot 8: Rankin (4hr arrivals)
ax8 = fig.add_subplot(4,4,8)
x = day_summary['Weekday']
y = day_summary['rankin_4hr'] 
ax8.plot(x,y)
ax8.set_xlabel('Day of week')
ax8.set_ylabel('Rankin (mean)')
ax8.set_title('Mean pre-stroke\nmodified Rankin\n(arrivals 4hrs from onset)')

# Subplot 9: NIHSS (4hr arrivals)
ax9 = fig.add_subplot(4,4,9)
x = day_summary['Weekday']
y = day_summary['nihss_4hr'] 
ax9.plot(x,y)
ax9.set_xlabel('Day of week')
ax9.set_ylabel('NIHSS (mean)')
ax9.set_title('Mean NIH Stroke Scale\n(arrivals 4hrs from onset)')

# Subplot 10: onset_arrival (4hr arrivals)
ax10 = fig.add_subplot(4,4,10)
x = day_summary['Weekday']
y = day_summary['onset_arrival'] 
ax10.plot(x,y)
ax10.set_xlabel('Day of week')
ax10.set_ylabel('Onset to arrival (minutes, mean)')
ax10.set_title('Mean onset to arrival\n(arrivals 4hrs from onset)')

# Subplot 11: scan_4hrs (4hr arrivals)
ax11 = fig.add_subplot(4,4,11)
x = day_summary['Weekday']
y = day_summary['scan_4hrs'] * 100
ax11.plot(x,y)
ax11.set_xlabel('Day of week')
ax11.set_ylabel('Proportion scanned within 4hrs (%)')
ax11.set_title('Proportion scanned within\n4hrs of arrival\n(arrivals 4hrs from onset)')

# Subplot 12: arrival_scan (4hr scan)
ax12 = fig.add_subplot(4,4,12)
x = day_summary['Weekday']
y = day_summary['arrival_scan'] 
ax12.plot(x,y)
ax12.set_xlabel('Day of week')
ax12.set_ylabel('Arrival to scan (minutes, mean)')
ax12.set_title('Mean arrival to scan\n(scanned 4hrs from onset)')

# Subplot 13: thrombolysis (4hr scan)
ax13 = fig.add_subplot(4,4,13)
x = day_summary['Weekday']
y = day_summary['thrombolyse_4hr']  * 100
ax13.plot(x,y)
ax13.set_xlabel('Day of week')
ax13.set_ylabel('Thrombolsyis(%)')
ax13.set_title('Thrombolysis use\n(scanned 4hrs from onset)')

# Subplot 14: scan_to_needle
ax14 = fig.add_subplot(4,4,14)
x = day_summary['Weekday']
y = day_summary['scan_to_needle'] 
ax14.plot(x,y)
ax14.set_xlabel('Day of week')
ax14.set_ylabel('Scan to needle (minutes, mean)')
ax14.set_title('Mean scan to needle')

# Subplot 15: arrival_to_needle
ax15 = fig.add_subplot(4,4,15)
x = day_summary['Weekday']
y = day_summary['arrival_to_needle'] 
ax15.plot(x,y)
ax15.set_xlabel('Day of week')
ax15.set_ylabel('Arrival to needle (minutes, mean)')
ax15.set_title('Mean arrival to needle')

# Subplot 16: onset_to_needle
ax16 = fig.add_subplot(4,4,16)
x = day_summary['Weekday']
y = day_summary['onset_to_needle'] 
ax16.plot(x,y)
ax16.set_xlabel('Day of week')
ax16.set_ylabel('Onset to needle (minutes, mean)')
ax16.set_title('Mean onset to needle')

# Save and show
plt.tight_layout(pad=2)
plt.savefig('output/stats_by_day_of_week.jpg', dpi=300)
plt.show();

## Observations

* Variation in thrombolysis use across the week is about 1.5 percentage points, with weekends having slightly higher thrombolysis use than weekdays.

* Generally patterns by weekday are weak when considered at a national average level.