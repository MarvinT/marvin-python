def savefig(g, name, folder='Figures/', format=None, formats=['png', 'pdf', 'svg', 'eps'], bbox_inches='tight', transparent=True, pad_inches=0):
	if format:
		formats = [format]
	for format in formats:
	    g.savefig(folder+name+'.'+format, format=format, bbox_inches=bbox_inches, transparent=transparent, pad_inches=pad_inches)