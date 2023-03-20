import sys
from time import time
import pandas as pd
from rdkit import Chem
import pickle

# s= time()
df1 = pd.read_csv("/home/tomerweiss/PBHs-design/data/db-474K-annotated.csv")
#                  usecols=['index', 'name', 'total energy', 'HOMO-LUMO gap/eV'])
# print(time()-s)
# print(sys.getsizeof(df)/1024/1024)
# print(df.columns)
# print(len(df))

dir = "/home/tomerweiss/PBHs-design/data/db-474K-xyz/"
df = pickle.load(open("/home/tomerweiss/PBHs-design/data/db-474K-all-data.pkl", "rb"))

print(len(df))


def my_filter(mol):
    dists = Chem.Get3DDistanceMatrix(mol) * Chem.GetAdjacencyMatrix(mol)
    max_dist = dists.max()
    return max_dist < 2.0


df = df[df["mol"].apply(my_filter)]
print(len(df))

for index, row in df.iterrows():
    # s = time()
    Chem.MolToMolFile(row.mol, dir + str(row["name"]) + ".pkl")
    # m = Chem.MolFromMolFile(dir + str(row['name']) + '.pkl')
    # print(time()-s)
    # Chem.MolToXYZFile(row.mol, dir + str(row['name']) + '.xyz')
    if index % 10000 == 0:
        print(index)

df = df.drop(columns=["mol"])
df.to_csv("/home/tomerweiss/PBHs-design/data/db-474K-filtered.csv")
