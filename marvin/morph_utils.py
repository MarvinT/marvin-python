import numpy as np
from ephys import core, rigid_pandas
import sklearn
import sklearn.linear_model
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
import marvin as m
import cPickle as Pickle


def parse_stim_id(df, stim_id='stim_id', end='end', morph_dim='morph_dim', morph_pos='morph_pos', lesser_dim='lesser_dim', greater_dim='greater_dim'):
    df[end] = df[stim_id].isin(list('abcdefghi'))
    df[morph_dim] = df[~df[end]][stim_id].str[0:2]
    df[morph_pos] = df[~df[end]][stim_id].str[2:].astype(int)
    df[lesser_dim] = df[~df[end]][morph_dim].str[0]
    df[greater_dim] = df[~df[end]][morph_dim].str[1]

stim_blacklist = ['G117-56']


def load_ephys(block_path, good_clusters=None, collapse_endpoints=False, shuffle_endpoints=False):
    assert not (collapse_endpoints and shuffle_endpoints)
    spikes = core.load_spikes(block_path)

    if good_clusters is not None:
        spikes = spikes[spikes.cluster.isin(good_clusters)]

    stims = rigid_pandas.load_acute_stims(block_path)

    for rec, rec_group in stims.groupby('recording'):
        try:
            rec_group['stim_name'].astype(float)
            print 'going to have to remove float stim recording ', rec
            spikes = spikes[spikes['recording'] != rec]
            stims = stims[stims['recording'] != rec]
        except:
            for bad_stim in stim_blacklist:
                if bad_stim in rec_group['stim_name'].values:
                    print 'going to have to remove stim recording ', rec, ' because of ', bad_stim
                    spikes = spikes[spikes['recording'] != rec]
                    stims = stims[stims['recording'] != rec]

    fs = core.load_fs(block_path)
    stims['stim_duration'] = stims['stim_end'] - stims['stim_start']
    rigid_pandas.timestamp2time(stims, fs, 'stim_duration')

    stim_ids = stims['stim_name']
    stim_ids = stim_ids.str.replace('_rec', '')
    stim_ids = stim_ids.str.replace('_rep\d\d', '')
    if collapse_endpoints:
        stim_ids = stim_ids.str.replace('[a-i]001', '')
        for motif in 'abcdefgh':
            stim_ids = stim_ids.str.replace('[a-i]%s128' % (motif), motif)
    stims['stim_id'] = stim_ids
    parse_stim_id(stims)

    if shuffle_endpoints:
        end_stims = stims[(stims['morph_pos'] == 1) |
                          (stims['morph_pos'] == 128)]
        for morph_pos, morph_pos_group in end_stims.groupby('morph_pos'):
            if morph_pos == 1:
                end_stims.loc[morph_pos_group.index,
                              'end_stim'] = morph_pos_group['morph_dim'].str[0]
            elif morph_pos == 128:
                end_stims.loc[morph_pos_group.index,
                              'end_stim'] = morph_pos_group['morph_dim'].str[1]

        for end_stim, end_stim_group in end_stims.groupby('end_stim'):
            stims.loc[end_stim_group.index, 'stim_id'] = end_stim_group[
                'stim_id'].values[np.random.permutation(len(end_stim_group))]

    rigid_pandas.count_events(stims, index='stim_id')

    spikes = spikes.join(rigid_pandas.align_events(spikes, stims,
                                                   columns2copy=['stim_id', 'morph_dim', 'morph_pos',
                                                                 'stim_presentation', 'stim_start', 'stim_duration']))

    spikes['stim_aligned_time'] = (spikes['time_samples'].values.astype('int') -
                                   spikes['stim_start'].values)
    rigid_pandas.timestamp2time(spikes, fs, 'stim_aligned_time')

    return spikes


def cluster_accuracy(cluster, cluster_group, morphs, max_num_reps, n_folds=10, n_dim=50, tau=.01, stim_length=.4):
    accuracies = pd.DataFrame(index=np.arange(len(morphs) * n_folds),
                              columns=['cluster', 'morph', 'i', 'accuracy'])
    idx = 0
    filtered_responses = {}
    for motif, motif_group in cluster_group.groupby('stim_id'):
        trial_groups = motif_group.groupby(['recording', 'stim_presentation'])
        filtered_responses[motif] = trial_groups['stim_aligned_time'].apply(
            lambda x: m.filtered_response(x.values, tau=tau))
    t = np.linspace(0, stim_length, n_dim)
    x = {}
    for motif in 'abcdefgh':
        x[motif] = np.zeros((max_num_reps, n_dim))
    for motif in filtered_responses:
        for i, fr in enumerate(filtered_responses[motif]):
            x[motif][i, :] = fr(t)

    for morph in morphs:
        l, r = morph
        x_concat = np.append(x[l], x[r], axis=0)
        y_concat = np.append(np.zeros(max_num_reps), np.ones(max_num_reps))
        for i, (train_idx, test_idx) in enumerate(StratifiedKFold(y_concat, n_folds=n_folds, shuffle=True)):
            model = LogisticRegression(solver='sag', warm_start=True)
            model.fit(x_concat[train_idx], y_concat[train_idx])
            y_test_hat = model.predict(x_concat[test_idx])
            accuracies.loc[idx] = [cluster, morph, i,
                                   np.mean(y_concat[test_idx] == y_test_hat)]
            idx += 1
    dtypes = {'cluster': 'int64', 'morph': 'str',
              'i': 'int64', 'accuracy': 'float64'}
    for col in dtypes:
        accuracies[col] = accuracies[col].astype(dtypes[col])
    return accuracies

block_paths = ['/mnt/cube/mthielk/analysis/B1101/kwik/Pen01_Lft_AP2500_ML750__Site02_Z1880__B1101_cat_P01_S02_1',
               '/mnt/cube/mthielk/analysis/B1101/kwik/Pen01_Lft_AP2500_ML750__Site03_Z2000__B1101_cat_P01_S03_2',
               '/mnt/cube/mthielk/analysis/B1101/kwik/Pen01_Lft_AP2500_ML750__Site04_Z2300__B1101_cat_P01_S04_3',
               '/mnt/cube/mthielk/analysis/B1101/kwik/Pen02_Lft_AP2501_ML500__Site02_Z2050__B1101_cat_P02_S02_1',
               '/mnt/cube/mthielk/analysis/B1101/kwik/Pen02_Lft_AP2501_ML500__Site03_Z2222__B1101_cat_P02_S03_2',
               '/mnt/cube/mthielk/analysis/B1101/kwik/Pen02_Lft_AP2501_ML500__Site04_Z2410__B1101_cat_P02_S04_3',
               '/mnt/cube/mthielk/analysis/B1101/kwik/Pen02_Lft_AP2501_ML500__Site05_Z3072__B1101_cat_P02_S05_4',
               '/mnt/cube/mthielk/analysis/B1218/kwik/Pen01_Lft_AP2500_ML160__Site02_Z2337__B1218_cat_P01_S02_1',
               '/mnt/cube/mthielk/analysis/B1218/kwik/Pen01_Lft_AP2500_ML160__Site04_Z2583__B1218_cat_P01_S04_2',
               '/mnt/cube/mthielk/analysis/B1218/kwik/Pen01_Lft_AP2500_ML160__Site05_Z2950__B1218_cat_P01_S05_3',
               '/mnt/cube/mthielk/analysis/B1218/kwik/Pen01_Lft_AP2500_ML160__Site06_Z3368__B1218_cat_P01_S06_4',
               '/mnt/cube/mthielk/analysis/B1218/kwik/Pen01_Lft_AP2500_ML160__Site06_Z3368__B1218_cat_P01_S06_5',
               '/mnt/cube/mthielk/analysis/B1218/kwik/Pen01_Lft_AP2500_ML160__Site07_Z3721__B1218_cat_P01_S07_6',
               '/mnt/cube/mthielk/analysis/B1218/kwik/Pen01_Lft_AP2500_ML160__Site08_Z4323__B1218_cat_P01_S08_7',
               '/mnt/cube/mthielk/analysis/B1134/kwik/Pen01_Lft_AP2500_ML500__Site02_Z3100__B1134_cat_P01_S02_1',
               '/mnt/cube/mthielk/analysis/B1134/kwik/Pen01_Lft_AP2500_ML500__Site03_Z3200__B1134_cat_P01_S03_1']


def load_cluster_accuracies():
    with open('all_accuracies.pkl', 'rb') as f:
        accuracies = Pickle.load(f)
    cluster_accuracies = {block_path: accuracies[block_path].groupby(
        'cluster').agg(np.mean).sort_values('accuracy') for block_path in accuracies}
    return accuracies, cluster_accuracies
