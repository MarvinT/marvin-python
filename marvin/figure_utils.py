def savefig(g, name, folder='Figures/', format='eps'):
    g.savefig(folder+name+'.'+format, format=format)