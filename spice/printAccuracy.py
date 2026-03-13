import pandas as pd
import os

print('Model,Overall,Small Ligands,Large Ligands,Peptides,Dimers,Neutral,Charged')
for name in os.listdir('.'):
    if name.endswith('csv') and name != 'summary.csv':
        df = pd.read_csv(name)
        ligands = df[~df.name.str.contains(' ') & ~df.name.str.contains('-')]
        print(name[:-4], df.error.mean(), ligands[ligands.atoms<60].error.mean(), ligands[ligands.atoms>60].error.mean(),
              df[df.name.str.contains('-')].error.mean(), df[df.name.str.contains(' ')].error.mean(),
              df[df.charge==0].error.mean(), df[df.charge!=0].error.mean(), sep=',')
