'''
Replicaty run_analysis.R
'''
import pickle
import os
import pystan
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


if __name__ == "__main__":
    compile_model = 0 # compile stan model
    run_fit = 0 # run fit
    do_save = 1 # save stan model, res,

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
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    axes.hlines(range(1,28), xlim[0], xlim[1], color=3*[0.8])
    axes.vlines(range(int(xlim[0]), int(xlim[1]+1)), ylim[0], ylim[1], color=3*[0.8])
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    fig.tight_layout()
    plt.savefig('figure1.png')

    #################
    # Model fitting #
    #################
    dat = {}
    dat['P'] = len(first_last_df)   # number of data points
    dat['N'] = len(test_final.index)    # number of patients
    dat['day_of_test'] = test_final['day'].values
    dat['test_result'] = (test_final['pcr_result'].values * 1.0).astype(int)
    dat['patient_ID'] = test_final['num_id'].values
    dat['time_first_symptom'] = first_last_df['first_symp_day'].values
    dat['time_last_asym'] = first_last_df['last_asym_day'].values
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081172/
    dat['lmean'] = 1.621 # EpiNow2::incubation_periods$mean
    dat['lsd'] = 0.418 # EpiNow2::incubation_periods$sd

    # Upper bound on time of infection, infection must occur before
    # first symptomatic report or first positive PCR, whichever
    # is first
    te_upper_bound = []
    for num_id, group in test_final.groupby('num_id'):
        pcr_positive = group['pcr_result'] == True
        if np.any(group[pcr_positive]['day'] < group[pcr_positive]['first_symp_day']):
            te_upper_bound.append(group[np.logical_and(pcr_positive, group['day'] < group['first_symp_day'])]['day'].min())
        else:
            # pass
            te_upper_bound.append(group['first_symp_day'].unique()[0])
    dat['te_upper_bound'] = te_upper_bound

    if (not os.path.exists('pcr_breakpoint.pkl')) or compile_model:
        ocode = open('pcr_breakpoint.stan', 'r').read()
        mod = pystan.StanModel(model_code=ocode)
        if do_save:
            pickle.dump(mod, open('pcr_breakpoint.pkl', 'wb'))
    else:
        mod = pickle.load(open('pcr_breakpoint.pkl', 'rb'))
    if run_fit or (not os.path.exists('res.pkl')):
        fit = mod.sampling(chains=4,
                           iter = 2000,
                           warmup = 1000,
                           data = dat,
                           control = {'adapt_delta': 0.9,
                                      'stepsize': 0.75,
                                      'max_treedepth': 13})
        res = fit.extract()
        if do_save:
            pickle.dump(res, open('res.pkl', 'wb'))
    else:
        res = pickle.load(open('res.pkl', 'rb'))


    ##########################################
    # Figure 2: Posterior of infection times #
    ##########################################

    allsamp = res['T_e']     # num_samples x P
    # days = np.arange(int(allsamp.min())-0.5, int(allsamp.max() + 1.5), 1)
    days = np.arange(int(allsamp.min()), int(allsamp.max() + 2), 1)
    fig, axes = plt.subplots(1,1,figsize=(12,8))
    for num_id in range(1, allsamp.shape[1]+1):
        n, _ = np.histogram(allsamp[:, num_id-1], days)
        n = n / n.sum()
        n *= 2
        axes.step(np.concatenate((days, [0])), num_id + np.concatenate(([0], n, [0])), color=colors[4], zorder=2)
        id = first_last_df[first_last_df['num_id'] == num_id]
        xra = np.array([id['last_asym_day'].values[0], id['first_symp_day'].values[0]])
        axes.errorbar(xra.mean(), num_id, xerr=np.abs(xra - xra.mean()).reshape(-1, 1),
                      fmt='', ecolor='g', color='g', capsize=5, zorder=4, elinewidth=2, capthick=2)
    axes.set_xlabel('Day')
    axes.set_ylabel('Participant ID')
    axes.set_xlim((datetime(2020, 3, 13) - start_date).days, (datetime(2020, 4, 19) - start_date).days)
    axes.set_ylim(0, 28)
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    axes.hlines(range(1,28), xlim[0], xlim[1], color=3*[0.8], zorder=3)
    axes.vlines(range(int(xlim[0]), int(xlim[1]+1)), ylim[0], ylim[1], color=3*[0.8])
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    fig.tight_layout()
    plt.savefig('figure2.png')
    print('done')
