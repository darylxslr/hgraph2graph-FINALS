import os
import sys
from rdkit import Chem
# Add the current directory to path so it can find the hgraph module
sys.path.append(os.getcwd())
from hgraph import MolGraph
import os
import sys
from rdkit import Chem

input_file = 'data/antibiotics/all.txt'
output_file = 'data/antibiotics/all_clean.txt'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found!")
    exit()

with open(input_file, 'r') as f:
    smiles_list = [line.strip().split()[0] for line in f if line.strip()]

clean_smiles = []
print(f"Stress-testing {len(smiles_list)} molecules against HGraph logic...")

for smi in smiles_list:
    try:
        # 1. Basic RDKit conversion
        mol = Chem.MolFromSmiles(smi)
        if mol is None: 
            continue
        
        # 2. Force Kekulization - this catches the "Can't kekulize" errors early
        Chem.Kekulize(mol, clearAromaticFlags=True)
        
        # 3. HGraph internal check 
        # This builds the junction tree; if this fails, the molecule is too complex
        test_graph = MolGraph(smi)
        
        # 4. Final check: Can we convert back to a clean string?
        # We use canonical SMILES here to ensure consistency
        final_smi = Chem.MolToSmiles(mol)
        clean_smiles.append(final_smi)
        
    except Exception as e:
        # If any part of the process fails, skip it
        # print(f"Skipping {smi} due to: {e}") 
        continue

with open(output_file, 'w') as f:
    for smi in clean_smiles:
        f.write(smi + '\n')

print(f"Done! {len(clean_smiles)} molecules are 100% safe for HGraph.")
# Add the current directory to path so it can find the hgraph module
sys.path.append(os.getcwd())

input_file = 'data/antibiotics/all.txt'
output_file = 'data/antibiotics/all_clean.txt'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found!")
    exit()

with open(input_file, 'r') as f:
    smiles_list = [line.strip().split()[0] for line in f if line.strip()]

clean_smiles = []
print(f"Stress-testing {len(smiles_list)} molecules against HGraph logic...")

for smi in smiles_list:
    try:
        # 1. Basic RDKit conversion
        mol = Chem.MolFromSmiles(smi)
        if mol is None: 
            continue
        
        # 2. Force Kekulization - this catches the "Can't kekulize" errors early
        Chem.Kekulize(mol, clearAromaticFlags=True)
        
        # 3. HGraph internal check 
        # This builds the junction tree; if this fails, the molecule is too complex
        test_graph = MolGraph(smi)
        
        # 4. Final check: Can we convert back to a clean string?
        # We use canonical SMILES here to ensure consistency
        final_smi = Chem.MolToSmiles(mol)
        clean_smiles.append(final_smi)
        
    except Exception as e:
        # If any part of the process fails, skip it
        # print(f"Skipping {smi} due to: {e}") 
        continue

with open(output_file, 'w') as f:
    for smi in clean_smiles:
        f.write(smi + '\n')

print(f"Done! {len(clean_smiles)} molecules are 100% safe for HGraph.")