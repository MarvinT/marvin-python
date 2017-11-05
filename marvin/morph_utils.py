import numpy as np
from ephys import core, rigid_pandas


def parse_stim_id(df, stim_id='stim_id', end='end', morph_dim='morph_dim', morph_pos='morph_pos', lesser_dim='lesser_dim', greater_dim='greater_dim'):
    df[end] = df[stim_id].isin(list('abcdefghi'))
    df[morph_dim] = df[~df[end]][stim_id].str[0:2]
    df[morph_pos] = df[~df[end]][stim_id].str[2:].astype(int)
    df[lesser_dim] = df[~df[end]][morph_dim].str[0]
    df[greater_dim] = df[~df[end]][morph_dim].str[1]


def load_ephys(block_path, good_clusters=None, collapse_endpoints=False, shuffle_endpoints=False):
    assert not (collapse_endpoints and shuffle_endpoints)
    spikes = core.load_spikes(block_path)

    if good_clusters is not None:
        spikes = spikes[spikes.cluster.isin(good_clusters)]

    stims = rigid_pandas.load_acute_stims(block_path)
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
