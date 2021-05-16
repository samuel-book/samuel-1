# Stroke pathway timing distribution

## Aim

Visualise distributions of timings for:

* Onset to arrival (when known)
* Arrival to scan
* Scan to needle

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import data
raw_data = pd.read_csv(
    './../data/2019-11-04-HQIP303-Exeter_MA.csv', low_memory=False)

# Set up figure
fig = plt.figure(figsize=(12,6))

# Subplot 1: Histogram of onset to arrival

onset_to_arrival = raw_data['S1OnsetToArrival_min']
# Limit to arricals within 8 hours
onset_to_arrival = raw_data['S1OnsetToArrival_min']
mask = onset_to_arrival <= 480
onset_to_arrival = onset_to_arrival[mask]

ax1 = fig.add_subplot(131)
bins = np.arange(0, 481, 10)
ax1.hist(onset_to_arrival, bins=bins, rwidth=1.0)
ax1.set_xlabel('Onset to arrival')
ax1.set_ylabel('Count')
ax1.set_title('Onset to arrival')
ax1.axes.get_yaxis().set_visible(False)


# Subplot 2: Histogram of arrival to scan

arrival_to_scan = raw_data['S2BrainImagingTime_min']
# Limit to arrivals within 4 hours
mask = arrival_to_scan <= 480
arrival_to_scan = arrival_to_scan[mask]

ax2 = fig.add_subplot(132)
bins = np.arange(0, 481, 10)
ax2.hist(arrival_to_scan, bins=bins, rwidth=1)
ax2.set_xlabel('Arrival to scan (mins)')
ax2.set_ylabel('Count')
ax2.set_title('Arrival to scan')
ax2.axes.get_yaxis().set_visible(False)

# Subplot 2: Histogram of scan to needle

scan_to_needle = \
    raw_data['S2ThrombolysisTime_min'] - raw_data['S2BrainImagingTime_min']

ax3 = fig.add_subplot(133)
bins = np.arange(0, 240, 5)
ax3.hist(scan_to_needle, bins=bins, rwidth=1)
ax3.set_xlabel('Scan to needle (mins)')
ax3.set_ylabel('Count')
ax3.set_title('Scan to needle')
ax3.axes.get_yaxis().set_visible(False)


# Save and show
plt.tight_layout(pad=2)
plt.savefig('output/pathway_distribution.jpg', dpi=300)
plt.show();

## Observations

* All timings show a right skew
* Choose log normal distributions for pathway process times