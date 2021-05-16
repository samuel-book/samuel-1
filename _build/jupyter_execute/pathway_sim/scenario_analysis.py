# Analysis of alternative pathway scenarios

## Load libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

## Load data

# Scenario results
results = pd.read_csv('./output/scenario_results.csv')
# Pathway performance paramters used in scenarios
performance_base = pd.read_csv('./output/performance_base.csv')
performance_speed = pd.read_csv('./output/performance_speed.csv')
performance_onset = pd.read_csv('./output/performance_onset.csv')
performance_speed_onset = pd.read_csv('./output/performance_speed_onset.csv')
performance_speed_benchmark = \
    pd.read_csv('./output/performance_speed_benchmark.csv')
performance_onset_benchmark = \
    pd.read_csv('./output/performance_onset_benchmark.csv')
performance_speed_onset_benchmark = \
    pd.read_csv('./output/performance_speed_onset_benchmark.csv')

performance_base

## Collate key results

Collate key results together in a DataFrame.

# Add admission numbers to results
admissions = performance_base[['stroke_team', 'admissions']]
results = results.merge(
    admissions, how='left', left_on='stroke_team', right_on='stroke_team')

# Calculate numbers thrombolysed
results['thrombolysed'] = \
    results['admissions'] * results['Percent_Thrombolysis_(mean)'] / 100

# Calculate additional good outcomes
results['add_good_outcomes'] = (results['admissions'] * 
    results['Additional_good_outcomes_per_1000_patients_(mean)'] / 1000)

# Get key results
key_results = pd.DataFrame()
key_results['stroke_team'] = results['stroke_team']
key_results['scenario'] = results['scenario']
key_results['admissions'] = results['admissions']
key_results['thrombolysis_rate'] = results['Percent_Thrombolysis_(mean)']
key_results['additional_good_outcomes_per_1000_patients'] = \
    results['Additional_good_outcomes_per_1000_patients_(mean)']
key_results['patients_receiving_thrombolysis'] = results['thrombolysed']
key_results['add_good_outcomes'] = results['add_good_outcomes']

key_results

key_results.to_csv('./output/key_scenario_results.csv')

## Overall results

columns = ['admissions', 'patients_receiving_thrombolysis', 'add_good_outcomes']
summary_stats = key_results.groupby('scenario')[columns].sum()
summary_stats['percent_thrombolysis'] = (100 *
    summary_stats['patients_receiving_thrombolysis'] / summary_stats['admissions'])
summary_stats['add_good_outcomes_per_1000'] = (1000 *
    summary_stats['add_good_outcomes'] / summary_stats['admissions'])

# Re-order
order = {'base': 1, 'speed': 2, 'onset': 3, 'benchmark': 4, 'speed_onset': 5,
    'speed_benchmark': 6, 'onset_benchmark':7, 'speed_onset_benchmark': 8}
df_order = [order[x] for x in list(summary_stats.index)]
summary_stats['order'] = df_order
summary_stats.sort_values('order', inplace=True)

# Select cols of interest
summary_stats = summary_stats[['percent_thrombolysis', 'add_good_outcomes_per_1000']]

base_thrombolysis = summary_stats.loc['base']['percent_thrombolysis']
summary_stats['Percent increase thrombolysis'] = (100 * (
        summary_stats['percent_thrombolysis'] / base_thrombolysis -1))

base_add_good_outcomes = summary_stats.loc['base']['add_good_outcomes_per_1000']
summary_stats['Percent increase good_outcomes'] = (100 * (
    summary_stats['add_good_outcomes_per_1000'] / base_add_good_outcomes -1))

summary_stats = summary_stats.round(2)

summary_stats.to_csv('./output/summary_net_results.csv')

summary_stats

fig = plt.figure(figsize=(10,7))

ax1 = fig.add_subplot(121)
x = list(summary_stats.index)
y1 = summary_stats['percent_thrombolysis'].values
ax1.bar(x,y1)
ax1.set_ylim(0,20)
plt.xticks(rotation=90)
plt.yticks(np.arange(0,22,2))
ax1.set_title('Thrombolysis use (%)')
ax1.set_ylabel('Thrombolysis use (%)')
ax1.grid(axis = 'y')

ax2 = fig.add_subplot(122)
x = list(summary_stats.index)
y1 = summary_stats['add_good_outcomes_per_1000'].values
ax2.bar(x,y1, color='r')
ax2.set_ylim(0,20)
plt.xticks(rotation=90)
plt.yticks(np.arange(0,22,2))
ax2.set_title('Additional good outcomes\nper 1,000 admissions')
ax2.set_ylabel('Additional good outcomes\nper 1,000 admissions')
ax2.grid(axis = 'y')

plt.tight_layout(pad=2)

plt.savefig('./output/global_change.jpg', dpi=300)

plt.show()




def compare_plot(base_rx, test_rx, base_benefit, test_benefit, name):
    
    # Set up sublot
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    
    # Thrombolysis use    
    x = base_rx
    y = test_rx
    for i in range(len(x)):
        if y[i] >= x[i]:
            ax[0].plot([x[i],x[i]],[x[i],y[i]],'g-o', alpha=0.6)
        else:
            ax[0].plot([x[i],x[i]],[x[i],y[i]],'r-o', alpha=0.6)
    ax[0].set_xlim(0)
    ax[0].set_ylim(0)
    ax[0].grid()
    ax[0].set_xlabel('Base thrombolysis use (%)')
    ax[0].set_ylabel('Scenario thrombolysis use (%)')
    ax[0].set_title('Thrombolysis use')
    
    # Benefit
    x = base_benefit
    y = test_benefit
    for i in range(len(x)):
        if y[i] >= x[i]:
            ax[1].plot([x[i],x[i]],[x[i],y[i]],'g-o', alpha=0.6)
        else:
            ax[1].plot([x[i],x[i]],[x[i],y[i]],'r-o', alpha=0.6)
    ax[1].set_xlim(0)
    ax[1].set_ylim(0)
    ax[1].grid()
    ax[1].set_xlabel('Base benefit')
    ax[1].set_ylabel(f'Scenario benefit')
    ax[1].set_title('Clinical benefit\n(additional good outcomes for 1,000 admissions)')
    
    
    # Make axes places consistent
    ax[0].xaxis.set_major_locator(MultipleLocator(5))
    ax[0].yaxis.set_major_locator(MultipleLocator(5))
    ax[1].xaxis.set_major_locator(MultipleLocator(5))
    ax[1].yaxis.set_major_locator(MultipleLocator(5))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    fig.tight_layout(pad=2)
    plt.savefig(f'{name}.jpg', dpi=300)

### Plot effect of speed

base_rx = key_results[key_results['scenario'] == 'base']['thrombolysis_rate'].values
scenario_rx = key_results[key_results['scenario'] == 'speed']['thrombolysis_rate'].values
base_benfit = key_results[key_results['scenario'] == 'base']['additional_good_outcomes_per_1000_patients'].values
scenario_benefit = key_results[key_results['scenario'] == 'speed']['additional_good_outcomes_per_1000_patients'].values


compare_plot(base_rx, scenario_rx, base_benfit, scenario_benefit,
                     './output/hospitals_speed')

### Plot effect of onset-known

base_rx = key_results[key_results['scenario'] == 'base']['thrombolysis_rate'].values
scenario_rx = key_results[key_results['scenario'] == 'onset']['thrombolysis_rate'].values
base_benfit = key_results[key_results['scenario'] == 'base']['additional_good_outcomes_per_1000_patients'].values
scenario_benefit = key_results[key_results['scenario'] == 'onset']['additional_good_outcomes_per_1000_patients'].values


compare_plot(base_rx, scenario_rx, base_benfit, scenario_benefit,
                     './output/hospitals_onset')

### Plot effect of benchmark decisions

base_rx = key_results[key_results['scenario'] == 'base']['thrombolysis_rate'].values
scenario_rx = key_results[key_results['scenario'] == 'benchmark']['thrombolysis_rate'].values
base_benfit = key_results[key_results['scenario'] == 'base']['additional_good_outcomes_per_1000_patients'].values
scenario_benefit = key_results[key_results['scenario'] == 'benchmark']['additional_good_outcomes_per_1000_patients'].values


compare_plot(base_rx, scenario_rx, base_benfit, scenario_benefit,
                     './output/hospitals_benchmark')

## Plot effect of all sceanrio changes

base_rx = key_results[key_results['scenario'] == 'base']['thrombolysis_rate'].values
scenario_rx = key_results[key_results['scenario'] == 'speed_onset_benchmark']['thrombolysis_rate'].values
base_benfit = key_results[key_results['scenario'] == 'base']['additional_good_outcomes_per_1000_patients'].values
scenario_benefit = key_results[key_results['scenario'] == 'speed_onset_benchmark']['additional_good_outcomes_per_1000_patients'].values


compare_plot(base_rx, scenario_rx, base_benfit, scenario_benefit,
                     './output/hospitals_speed_onset_benchmark')

### Histogram of shift in distribution of thrombolysis use and benefit

base_rx = key_results[key_results['scenario'] == 'base']['thrombolysis_rate'].values
scenario_rx = key_results[key_results['scenario'] == 'speed_onset_benchmark']['thrombolysis_rate'].values
base_benfit = key_results[key_results['scenario'] == 'base']['additional_good_outcomes_per_1000_patients'].values
scenario_benefit = key_results[key_results['scenario'] == 'speed_onset_benchmark']['additional_good_outcomes_per_1000_patients'].values


fig, ax = plt.subplots(1,2,figsize=(10,6))
bins = np.arange(1,32)
ax[0].hist(base_rx, bins = bins, color='k', linewidth=2,
           label='control', histtype='step')
ax[0].hist(scenario_rx, bins = bins, color='0.7', linewidth=2,
           label='all in-hospital changes', histtype='stepfilled')
ax[0].grid()
ax[0].set_xlabel('Thrombolysis use (%)')
ax[0].set_ylabel('count')
ax[0].set_ylim(0,30)
ax[0].yaxis.set_major_locator(MultipleLocator(5))
ax[0].set_title('Thrombolysis use')
ax[0].legend()

ax[1].hist(base_benfit, bins = bins, color='k', linewidth=2,
           label='control', histtype='step')
ax[1].hist(scenario_benefit, bins = bins, color='0.7', linewidth=2,
           label='all in-hospital changes', histtype='stepfilled')
ax[1].grid()
ax[1].set_xlabel('Clincial benefit')
ax[1].set_ylabel('count')
ax[1].set_ylim(0,26)
ax[1].yaxis.set_major_locator(MultipleLocator(5))
ax[1].set_title('Clincial benefit')
ax[1].legend()

plt.savefig('./output/histograms.jpg', dpi=300)

## Bar charts of individual changes at all hospitals

Here we summarise the effect of individual changes at each hospital.

Note that actual combined changes will be more than additive, but these plots give an indication of what the most significant effects will be across all hospitals.

# Pivot results by scenario type
results_pivot = key_results.pivot(index='stroke_team', columns='scenario')

hosp_per_chart = np.ceil(results_pivot.shape[0]/2)

# Thrombolysis chart
fig, axs = plt.subplots(2,1, figsize=(12,12), sharey=True)
# 4 subplots
i=0
for ax in axs.flat:
    # Get subgroup of data for plot
    start = int(hosp_per_chart * i)
    end = int(hosp_per_chart * (i + 1))
    subgroup = results_pivot.iloc[start:end]
    # Get effect of speed (avoid negatives)
    speed = subgroup['thrombolysis_rate']['speed'] - subgroup['thrombolysis_rate']['base']
    speed = list(map (lambda x: max(0,x), speed))
    # Get effect of known onset (avoid negatives)
    onset = subgroup['thrombolysis_rate']['onset'] - subgroup['thrombolysis_rate']['base']
    onset = list(map (lambda x: max(0,x), onset))
    # Get effect of decision (avoid negatives)
    eligible = subgroup['thrombolysis_rate']['benchmark'] - subgroup['thrombolysis_rate']['base']
    eligible = list(map (lambda x: max(0,x), eligible))
    
    x = range(start, start + subgroup.shape[0])
    ax.bar(x, speed, color='b', label = 'Speed')
    ax.bar(x, onset, color='g', bottom = speed, label = 'Onset known')
    ax.bar(x, eligible, color='r', bottom = np.array(speed) + np.array(onset),
           label = 'Benchmark decision')
    ax.legend(loc='upper right')
    ax.set_xlabel('Hospital')
    ax.set_ylabel('Increase in thrombolysis use (percentage points)')
    # Put y tick label son all charts
    ax.yaxis.set_tick_params(which='both', labelbottom=True)
    
    i += 1
      
plt.tight_layout(pad=2)
plt.savefig('./output/all_hosp_bar_thrombolysis.jpg', dpi=300)
plt.show()

hosp_per_chart = np.ceil(results_pivot.shape[0]/2)

# Outcomes chart
fig, axs = plt.subplots(2,1, figsize=(12,12), sharey=True)
# 4 subplots
i=0
for ax in axs.flat:
    # Get subgroup of data for plot
    start = int(hosp_per_chart * i)
    end = int(hosp_per_chart * (i + 1))
    subgroup = results_pivot.iloc[start:end]
    
    # Get effect of speed (avoid negatives)
    speed = subgroup['additional_good_outcomes_per_1000_patients']['speed'] - \
        subgroup['additional_good_outcomes_per_1000_patients']['base']
    speed = list(map (lambda x: max(0,x), speed))
    # Get effect of known onset (avoid negatives)
    onset = subgroup['additional_good_outcomes_per_1000_patients']['onset'] - \
        subgroup['additional_good_outcomes_per_1000_patients']['base']
    onset = list(map (lambda x: max(0,x), onset))
    # Get effect of decision (avoid negatives)
    eligible = subgroup['additional_good_outcomes_per_1000_patients']['benchmark'] - \
        subgroup['additional_good_outcomes_per_1000_patients']['base']
    eligible = list(map (lambda x: max(0,x), eligible))
    
    x = range(start, start + subgroup.shape[0])
    ax.bar(x, speed, color='b', label = 'Speed')
    ax.bar(x, onset, color='g', bottom = speed, label = 'Onset known')
    ax.bar(x, eligible, color='r', bottom = np.array(speed) + np.array(onset),
           label = 'Benchmark decision')
    ax.legend(loc='upper right')
    ax.set_xlabel('Hospital')
    ax.set_ylabel('Increase in good outcomes (per 1,000 admissions)')
    # Put y tick label son all charts
    ax.yaxis.set_tick_params(which='both', labelbottom=True)
    
    i += 1
      
plt.tight_layout(pad=2)
plt.savefig('./output/all_hosp_bar_outcomes.jpg', dpi=300)
plt.show()

## Results for individual hospitals

We may plot more detailed results at an individual hospital level.

def plot_hospital(data, id):
    
    hospital_data = data.iloc[id]
    
    max_val = max(hospital_data['thrombolysis_rate'].max(),
                  hospital_data['additional_good_outcomes_per_1000_patients'].max())
    
    max_val = 5 * int(max_val/5) + 5
    
    team = hospital_data.name
    
    # Sort results
    
    df = pd.DataFrame()
    df['thrombolysis_rate'] = hospital_data['thrombolysis_rate']
    df['outcomes'] = hospital_data['additional_good_outcomes_per_1000_patients']
    order = {'base': 1, 'speed': 2, 'onset': 3, 'benchmark': 4, 'speed_onset': 5,
    'speed_benchmark': 6, 'onset_benchmark':7, 'speed_onset_benchmark': 8}
    df_order = [order[x] for x in list(df.index)]
    df['order'] = df_order
    df.sort_values('order', inplace=True)   
    

    fig = plt.figure(figsize=(10,7))

    ax1 = fig.add_subplot(121)
    x = df['thrombolysis_rate'].index
    y1 = df['thrombolysis_rate']
    ax1.bar(x,y1)
    plt.xticks(rotation=90)
    ax1.set_title('Thrombolysis use (%)')
    ax1.set_ylabel('Thrombolysis use (%)')
    ax1.set_ylim(0, max_val)
    ax1.grid(axis = 'y')

    ax2 = fig.add_subplot(122)
    y1 = df['outcomes']
    ax2.bar(x,y1, color='r')
    plt.xticks(rotation=90)
    ax2.set_title('Additional good outcomes\nper 1,000 admissions')
    ax2.set_ylabel('Additional good outcomes\nper 1,000 admissions')
    ax2.set_ylim(0, max_val)
    ax2.grid(axis = 'y')
    
    plt.suptitle(f'Scenario results for team: {team}')

    plt.tight_layout(pad=2)
    
    plt.savefig(f'./output/hosp_results_{team}.jpg', dpi=300)
    
    plt.show()

An example where speed makes most difference.

plot_hospital(results_pivot, 103)

An example where determining stroke onset time makes most difference.

plot_hospital(results_pivot, 64)

An example where applying benchmark decion-making makes most difference.

plot_hospital(results_pivot, 54)