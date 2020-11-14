'''
Replicaty run_analysis.R
'''
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Load in participant data
test_final = pd.read_csv('test_data.csv')
symp_final = pd.read_csv('symptom_data.csv')
for index, row in test_final.iterrows():
    test_final.loc[index, 'date'] = datetime.strptime(row['date'], '%Y-%m-%d')
for index, row in symp_final.iterrows():
    symp_final.loc[index, 'date'] = datetime.strptime(row['date'], '%Y-%m-%d')

# Set days relative to a chosen start date
start_date = datetime(2020, 1, 1)
for index, row in test_final.iterrows():
    test_final.loc[index, 'day'] = int((row['date'] - start_date).days)
for index, row in test_final.iterrows():
    if isinstance(row['serology_date'], str):
        test_final.loc[index, 'serology_day'] = (datetime.strptime(row['serology_date'], '%Y-%m-%d') - start_date).days

## Add initial asymptomatic reports on enrollment day
for index, row in symp_final.iterrows():
    symp_final.loc[index, 'day'] = int((row['date'] - start_date).days)
for num_id, group in test_final.groupby('num_id'):
    symp_final = symp_final.append(pd.DataFrame([num_id, group['day'].min(), False], index=['num_id', 'day', 'symptom']).T,
                                   ignore_index=True)


## Find first symptomatic report dates and last asymptomatic report dates
first_last_df = pd.DataFrame(3*[], index=['num_id', 'first_symp_day', 'last_asym_day']).T
for num_id, group in symp_final.groupby('num_id'):
    first_symp = group[group['symptom'] == True]['day'].min()
    last_asym = group[np.logical_and(group['symptom'] == False, group['day'] < first_symp)]['day'].max()
    first_last_df = first_last_df.append(pd.DataFrame([num_id, first_symp, last_asym], index=['num_id', 'first_symp_day', 'last_asym_day']).T, ignore_index=True)
first_last_df['num_id'] = first_last_df['num_id'].astype('int')

# merge
test_final = pd.merge(test_final, first_last_df, on='num_id')

######################################
# Figure 1: Symptom and testing data #
######################################
dfy = pd.merge(symp_final, test_final, on=['num_id', 'day'])

fig, axes = plt.subplots(1,1,figsize=(12,8))
for num_id in range(1, 28):
    id = dfy[dfy['num_id'] == num_id]
    x0 = id['last_asym_day'].unique()[0]
    # plot symptoms
    x = id[id['symptom'] == True]['day'].values
    axes.plot(x - x0, len(x)*[num_id], 'o', color='r')
    # plot PCR negative results
    x = id[id['pcr_result'] == False]['day'].values
    axes.plot(x - x0, len(x)*[num_id], 'o', color='k', markerfacecolor='none')
    # plot PCR positive results
    x = id[id['pcr_result'] == True]['day'].values
    axes.plot(x - x0, len(x)*[num_id], 'o', color='g', markerfacecolor='none', markeredgewidth=1.5)
    # Serology result
    x = id[np.logical_and(np.isfinite(id['serology_day']), id['serology_day'] < id['first_symp_day'])]['serology_day'].values
    if len(x) > 0:
        axes.plot(x[0] - x0, [num_id], 'x', color='k')

axes.set_xlabel('Day')
axes.set_ylabel('Participant ID')
axes.set_ylim(0, 28)
axes.grid(True, which='minor')
xlim = axes.get_xlim()
ylim = axes.get_ylim()
axes.hlines(range(1,28), xlim[0], xlim[1], color=3*[0.8])
axes.vlines(range(int(xlim[0]), int(xlim[1]+1)), ylim[0], ylim[1], color=3*[0.8])
axes.set_xlim(xlim)
axes.set_ylim(ylim)
fig.tight_layout()
plt.savefig('figure1.png')