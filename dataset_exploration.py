import pandas as pd
import deepchem as dc
from rdkit import Chem
import numpy as np

print(">>> read the data file ... ")
df = pd.read_csv('datasets/clintox.csv.gz')
print("### Clintox Dataset")
print(">>> Number of molecules =", df.shape[0])
print(">>> Number of Tasks =", df.shape[1] - 1)
max_num_atoms = -1
for z in df['smiles']:
    if len(z)>max_num_atoms:
        max_num_atoms = len(z)

print(">>> Max Number of Atoms in a Molecule =", max_num_atoms)

for i in df.columns[1:]:
    print(df[i].value_counts())

print()
print(">>> read the data file ... ")
df = pd.read_csv('datasets/sider.csv.gz')
print("### Sider Dataset")
print(">>> Number of molecules =", df.shape[0])
print(">>> Number of Tasks =", df.shape[1] - 1)
max_num_atoms = -1
for z in df['smiles']:
    if len(z)>max_num_atoms:
        max_num_atoms = len(z)

print(">>> Max Number of Atoms in a Molecule =", max_num_atoms)

for i in df.columns[1:]:
    print(df[i].value_counts())

print()
print(">>> read the data file ... ")
df = pd.read_csv('datasets/toxcast_data.csv.gz')
print("### Toxcast Dataset")
print(">>> Number of molecules =", df.shape[0])
print(">>> Number of Tasks =", df.shape[1] - 1)
max_num_atoms = -1
for z in df['smiles']:
    if len(z)>max_num_atoms:
        max_num_atoms = len(z)

print(">>> Max Number of Atoms in a Molecule =", max_num_atoms)

# for i in df.columns[1:]:
#     print(df[i].value_counts())

print()
print(">>> read the data file ... ")
df = pd.read_csv('datasets/HIV.csv')
print("### HIV Dataset")
print(">>> Number of molecules =", df.shape[0])
print(">>> Number of Tasks =", df.shape[1] - 1)
max_num_atoms = -1
for z in df['smiles']:
    if len(z)>max_num_atoms:
        max_num_atoms = len(z)

print(">>> Max Number of Atoms in a Molecule =", max_num_atoms)

for i in df.columns[1:]:
    print(df[i].value_counts())
