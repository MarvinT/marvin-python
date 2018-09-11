import numpy as np
import pandas as pd
import scipy as sp
import sklearn as skl
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as plt
import seaborn as sns
import itertools
from joblib import Parallel, delayed
import marvin as m

def gen_dim_map(behavior_subj, psychometric_params, shuffle=False):
    dims = psychometric_params[behavior_subj].keys()
    if shuffle:
        return {dim:target for dim, target in zip(dims, np.random.permutation(dims))}
    else:
        return {dim:dim for dim in dims}
    
def in_pattern(string):
    return r'[' + string + r']'

def parse_effective_morph(df, behavior_subj):
    left, right = m.morph.training[behavior_subj].lower().split('|')
    
    df['inverted'] = df['lesser_dim'].str.match(in_pattern(right)) & df['greater_dim'].str.match(in_pattern(left))
    
    df['effective_dim'] = df['morph_dim']
    df.loc[df['inverted'], 'effective_dim'] = df['morph_dim'].str[::-1][df['inverted']]
    
    df['effective_pos'] = df['morph_pos']
    df.loc[df['inverted'], 'effective_pos'] = 128 - (df[df['inverted']]['morph_pos'] - 1)

def make_label_df(labels, behavior_subj, psychometric_params):
    label_df = pd.DataFrame(data={'stim_id':labels})
    m.morph.parse_stim_id(label_df)
    
    label_df['behave_data'] = False
    for dim, dim_group in label_df.groupby('morph_dim'):
        if dim in psychometric_params[behavior_subj]:
            label_df.loc[dim_group.index, 'behave_data'] = True
    
    parse_effective_morph(label_df, behavior_subj)
    return label_df

def make_behavior_df(behavior_subj, psychometric_params):
    morph_dims, morph_poss = zip(*itertools.product(psychometric_params[behavior_subj].keys(), np.arange(1, 129)))
    behavior_df = pd.DataFrame(data={'morph_dim': morph_dims, 'morph_pos': morph_poss})
    behavior_df['lesser_dim'] = behavior_df['morph_dim'].str[0]
    behavior_df['greater_dim'] = behavior_df['morph_dim'].str[1]
    parse_effective_morph(behavior_df, behavior_subj)
    for dim, dim_group in behavior_df.groupby('morph_dim'):
        psyc = m.normalized_four_param_logistic(psychometric_params[behavior_subj][dim])
        behavior_df.loc[dim_group.index, 'p_greater'] = dim_group['morph_pos'].apply(psyc)
    behavior_df['p_lesser'] = 1.0 - behavior_df['p_greater']
    behavior_df['p_left'], behavior_df['p_right'] = behavior_df['p_lesser'], behavior_df['p_greater']
    behavior_df.loc[behavior_df['inverted'], 'p_right'] = behavior_df.loc[behavior_df['inverted'], 'p_lesser']
    behavior_df.loc[behavior_df['inverted'], 'p_left'] = behavior_df.loc[behavior_df['inverted'], 'p_greater']
    return behavior_df

def shuffle_effective_dim(df, shuffle=False):
    if shuffle:
        behave_dims = df[df['behave_data']]['effective_dim'].unique()
        non_behave_dims = set(df['effective_dim'].unique()) - set(behave_dims)
        dim_map = {dim:target for dim, target in zip(behave_dims, np.random.permutation(behave_dims))}
        dim_map.update({dim:dim for dim in non_behave_dims})
        df['shuffled_dim'] = df['effective_dim'].map(dim_map)
    else:
        df['shuffled_dim'] = df['effective_dim']

def hold_one_out_psychometric_fit_dist(representations, labels, behavior_subj, psychometric_params,
                                       shuffle_count=1024, parallel=True, n_jobs=12):
    '''
    fits behavioral psychometric curves using the representation in a hold one out manner
    
    Parameters
    -----
    representations : np.array
        size = (num_data_points, num_dimensions)
    labels : iterable of string labels or np array of dtype='S5'
        labels : Pandas.DataFrame
            len = num_data_points
            required columns = ['morph_dim', 'morph_pos']
            overwritten/created columns = ['behave_data', 'p_r', 'p_l']
    behavior_subj : str
    shuffle_count : int
    calibrate : boolean
    
    Returns
    -----
    '''
    label_df = make_label_df(labels, behavior_subj, psychometric_params)
    behavior_df = make_behavior_df(behavior_subj, psychometric_params)
    
    if parallel:
        all_samples = Parallel(n_jobs=n_jobs)(delayed(calc_samples)(representations, label_df, behavior_df, idx, shuffle=shuffle) for idx, shuffle in [(i, i!=0) for i in xrange(shuffle_count+1)])
    else:
        all_samples = [calc_samples(representations, label_df, behavior_df, idx, shuffle=shuffle) for idx, shuffle in [(i, i!=0) for i in xrange(shuffle_count+1)]]
    all_samples_df = pd.concat(all_samples, ignore_index=True)
    all_samples_df['subj'] = behavior_subj
    return all_samples_df

def hold_one_out_psychometric_fit_dist_all_subj(representations, labels, psychometric_params,
                                                shuffle_count=1024, parallel=True, n_jobs=12):
    all_samples = []
    for subj in psychometric_params:
        print subj
        all_samples.append(hold_one_out_psychometric_fit_dist(representations, labels, subj, psychometric_params,
                                                              shuffle_count=shuffle_count, parallel=parallel,
                                                              n_jobs=n_jobs))
    return pd.concat(all_samples)

def merge_shuffle_df(label_df, behavior_df, shuffle=False):
    shuffle_effective_dim(label_df, shuffle=shuffle)
    shuffle_effective_dim(behavior_df, shuffle=False)
    return pd.merge(label_df, behavior_df[['shuffled_dim', 'effective_pos', 'p_left', 'p_right']], 
                        on=['shuffled_dim', 'effective_pos'], how='left', validate='m:1')

def calc_samples(representations, label_df, behavior_df, idx, shuffle=False, tol=1e-4):
    error_list, dim_list = fit_held_outs(merge_shuffle_df(label_df, behavior_df, shuffle=shuffle),
                               representations, tol=tol)
    return pd.DataFrame(data={'errors':error_list, 'held_out_dim':dim_list, 'shuffle_index':idx, 'shuffled':shuffle})

def fit_held_outs(merged_df, representations, accum='sse', tol=1e-4):
    mbdf = merged_df[merged_df['behave_data']]
    error_list = []
    dim_list = []
    for held_out_dim in mbdf['shuffled_dim'].unique():
        training_df = mbdf[mbdf['shuffled_dim'] != held_out_dim]
        held_out_df = mbdf[mbdf['shuffled_dim'] == held_out_dim]
        train_x = np.concatenate([representations[training_df.index,:], representations[training_df.index,:]])
        train_y = np.repeat([0, 1], len(training_df))
        train_weights = np.concatenate([training_df['p_left'], training_df['p_right']])

        test_x = representations[held_out_df.index,:]
        test_y = held_out_df['p_right']

        model = LogisticRegression(penalty='l2', tol=tol, warm_start=True).fit(train_x, train_y, sample_weight=train_weights)
        predicted_values = model.predict_proba(test_x)[:,1]
        
        dim_list.append(held_out_dim)
        if accum == 'df':
            fit_df = held_out_df[['stim_id', 'p_right']].copy()
            fit_df['predicted'] = predicted_values
            error_list.append(fit_df)
        elif accum == 'mse':
            error_list.append(np.square(predicted_values - test_y).mean())
        elif accum == 'sse':
            error_list.append(np.square(predicted_values - test_y).sum())
        elif accum == 'sigmoid fit':
            raise NotImplementedError
        else:
            raise Exception('invalid accum option')
    return error_list, dim_list

def gen_held_out_df(merged_df, representations, melt=False):
    held_out_df = pd.concat(fit_held_outs(merged_df, representations, accum='df')[0])
    if melt:
        held_out_df = pd.melt(held_out_df, id_vars=['stim_id'], value_vars=['p_right', 'predicted'],
                              var_name='legend', value_name='p_right')
    m.morph.parse_stim_id(held_out_df)
    return held_out_df

def plot_held_out(labels, representations, behavior_subj, psychometric_params):
    label_df = make_label_df(labels, behavior_subj, psychometric_params)
    print 'labeled'
    behavior_df = make_behavior_df(behavior_subj, psychometric_params)
    print 'behavior_df'
    merged_df = merge_shuffle_df(label_df, behavior_df)
    print 'merged'
    held_out_df = gen_held_out_df(merged_df, representations, melt=True)
    print 'held_out ... now plotting'
    row_order = held_out_df['lesser_dim'].unique()
    col_order = held_out_df['greater_dim'].unique()
    row_order.sort()
    col_order.sort()
    g = sns.lmplot(x='morph_pos', y='p_right', hue='legend', col='greater_dim', row='lesser_dim', 
               data=held_out_df, fit_reg=False, row_order=row_order, col_order=col_order)
    g = g.set_titles("{row_name}{col_name}")
    
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle(behavior_subj)
    return held_out_df

def shuffle_ks_df(samples_df):
    shuffled_grouped = samples_df.groupby('shuffled')
    shuffled = shuffled_grouped.get_group(True)['errors'].values
    fit = shuffled_grouped.get_group(False)['errors'].values

    for was_shuffled, shuffle_group in shuffled_grouped:
        grouped = shuffle_group.groupby(['shuffle_index', 'subj'])
        temp_results = np.zeros(len(grouped))
        temp_subj_results = ['' for i in xrange(len(grouped))]
        for i, ((shuffle_index, subj), group) in enumerate(grouped):
            temp_results[i] = sp.stats.mstats.ks_twosamp(group['errors'].values, shuffled, alternative='greater')[0]
            temp_subj_results[i] = subj

        df  = pd.DataFrame(columns=['subj', 'ks_stat'])
        df['subj'] = temp_subj_results
        df['ks_stat'] = temp_results

        if was_shuffled:
            shuffle_df = df
        else:
            unshuffle_df = df

    return shuffled_df, unshuffle_df

def merge_ks_dfs(shuffled_df, unshuffle_df, reset_index=True):
    temp_df = shuffled_df.merge(unshuffle_df, on=('subj'), suffixes=('_shuffled', '_unshuffled'))
    temp_df['p'] = temp_df['ks_stat_shuffled'] > temp_df['ks_stat_unshuffled']
    ks_df = temp_df.groupby('subj').apply(np.mean)
    if reset_index:
        ks_df = ks_df.reset_index()
    del ks_df['ks_stat_shuffled']
    return ks_df

def plot_ks_null_dist(shuffled_df, unshuffle_df, combined_subj_plot=False):
    if combined_subj_plot:
        f = plt.figure(figsize=(10,10))
        ax = f.gca()
    ks_df = merge_ks_dfs(shuffled_df, unshuffle_df, reset_index=False)
    for subj, subj_group in shuffled_df.groupby('subj'):
        if not combined_subj_plot:
            f = plt.figure(figsize=(10,10))
            ax = f.gca()
        print subj, float(np.sum(subj_group['ks_stat'] > ks_df.loc[subj, 'ks_stat_unshuffled'])) / len(subj_group)
        sns.distplot(subj_group['ks_stat'], norm_hist=True, label=subj, color=m.morph.behave_color_map[subj])
        plt.axvline(x=ks_df.loc[subj, 'ks_stat_unshuffled'], color=m.morph.behave_color_map[subj])
        ax.legend();
        ax.set_xlim([0, ax.get_xlim()[1]]);
        ax.set_title('ks stat null dist by subj');

def logistic_dim_discriminants(X, labels):
    dim_discriminants = {}
    labels = pd.Series(labels)
    morph_dims = labels.str[:2].unique()
    stim_ids, _ = m.morph.separate_endpoints(labels)
    motif_map = pd.DataFrame(stim_ids, columns=['motif']).groupby('motif')

    for morph_dim in morph_dims:
        lesser_dim, greater_dim = morph_dim
        endpoints_data = np.concatenate([X[motif_map.get_group(dim).index, :] for dim in morph_dim])
        endpoints_label = np.concatenate([np.ones_like(motif_map.get_group(dim).index) * (dim == morph_dim[1]) for dim in morph_dim])
        model = LogisticRegression(penalty='l2')
        model.fit(endpoints_data, endpoints_label)
        dim_discriminants[morph_dim] = model.coef_

    return dim_discriminants

def logistic_dim_reduction(X, labels):
    dim_discriminants = logistic_dim_discriminants(X, labels)
    proj_matrix = np.array([dim_discriminants[dim] / np.linalg.norm(dim_discriminants[dim]) for dim in dim_discriminants]).squeeze().T
    return X.dot(proj_matrix)