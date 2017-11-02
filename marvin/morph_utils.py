def parse_stim_id(df, stim_id='stim_id', morph_dim='morph_dim', morph_pos='morph_pos'):
    df[morph_dim] = df[~df[stim_id].str.match('^[abcdefghi]$')][stim_id].str[0:2]
    df[morph_pos] = df[~df[stim_id].str.match('^[abcdefghi]$')][stim_id].str[2:].astype(int)
