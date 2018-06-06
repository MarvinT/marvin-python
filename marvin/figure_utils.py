import seaborn as sns
import numpy as np

def gen_cmaps(num_maps, order=None, per_color=4, lpadding=4, rpadding=4):
    if not order:
        order = xrange(num_maps)
    per_color += np.zeros(num_maps, dtype=int)
    colors = sum([sns.blend_palette(['white', sns.hls_palette(num_maps, l=.65, s=1)[i], 'black'], n_colors=lpadding+npc+rpadding)[lpadding:lpadding+npc] 
                     for i, npc in zip(order, per_color)], [])
    return np.array(colors).reshape((np.sum(per_color),4))

def savefig(g, name, folder='Figures/', format=None, formats=['png', 'pdf', 'svg', 'eps'], bbox_inches='tight', transparent=True, pad_inches=0):
	if format:
		formats = [format]
	for format in formats:
	    g.savefig(folder+name+'.'+format, format=format, bbox_inches=bbox_inches, transparent=transparent, pad_inches=pad_inches)
