def savefig(g, name, folder='Figures/', format='pdf'):
    g.savefig(folder+name+'.'+format, format=format, bbox_inches='tight', transparent=True, pad_inches=0)