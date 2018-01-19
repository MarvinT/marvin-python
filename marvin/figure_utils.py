def savefig(g, name, folder='Figures/', format='pdf', transparent=True):
    g.savefig(folder+name+'.'+format, format=format, bbox_inches='tight', transparent=transparent, pad_inches=0)