import requests
import pandas as pd
import time
import os

# Create data directory
os.makedirs('data/antibiotics', exist_ok=True)

def check_patent_by_smiles(smiles):
    """Checks PubChem for patents using a SMILES string with headers."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # Step 1: Find CID by SMILES
        res = requests.post(f"{base_url}/compound/smiles/cids/JSON", 
                            data={'smiles': smiles}, 
                            headers=headers, timeout=10).json()
        cid = res['IdentifierList']['CID'][0]

        # Step 2: Check for Patents
        patent_req = requests.get(f"{base_url}/compound/cid/{cid}/xrefs/Patent/JSON", 
                                  headers=headers, timeout=10).json()
        return 1 if 'InformationList' in patent_req else 0
    except Exception:
        return 0

# 1. Load your existing ChEMBL data
print("Reading local ChEMBL data from data/chembl/all.txt...")
try:
    with open('data/chembl/all.txt', 'r') as f:
        # Taking 300 molecules for a balanced test
        all_smiles = [line.strip() for line in f.readlines()[:300]]
except FileNotFoundError:
    print("Error: Could not find data/chembl/all.txt. Check your folder structure!")
    exit()

# 2. Check patents for each
results = []
print(f"Checking patents for {len(all_smiles)} molecules. This will take ~3 minutes...")

for i, smiles in enumerate(all_smiles):
    is_patented = check_patent_by_smiles(smiles)
    results.append({"id": i, "smiles": smiles, "is_patented": is_patented})
    
    if i % 10 == 0:
        print(f"Progress: {i}/{len(all_smiles)} molecules checked...")
    
    time.sleep(0.4) # Crucial: prevents PubChem from blocking your IP

# 3. Save files
df = pd.DataFrame(results)
df.to_csv('data/antibiotics/metadata.csv', index=False)
df['smiles'].to_csv('data/antibiotics/all.txt', index=False, header=False)

print(f"\nSUCCESS! Files created:")
print(f"- data/antibiotics/all.txt (For training)")
print(f"- data/antibiotics/metadata.csv (For evaluation diagnostics)")