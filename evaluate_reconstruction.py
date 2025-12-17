import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from hgraph.mol_graph import MolGraph
from hgraph.vocab import PairVocab, common_atom_vocab
from hgraph.hgnn import HierVAE

# --- GPU/CPU SAFETY ---
if not torch.cuda.is_available():
    def cuda_fake(self, *args, **kwargs): return self
    torch.Tensor.cuda = cuda_fake
    torch.nn.Module.cuda = cuda_fake
    DEVICE = 'cpu'
else:
    DEVICE = 'cuda'

# --- PATHS ---
MODEL_PATH = 'ckpt/antibiotic-model/model.ckpt.440'
VOCAB_PATH = 'data/antibiotics/vocab.txt'
DATA_PATH = 'data/antibiotics/all_clean.txt'
OUTPUT_FILE = 'results.csv'
PLOT_FILE = 'tanimoto_histogram.png'

# 1. Setup Vocab
with open(VOCAB_PATH) as f:
    vocab_lines = [x.strip("\r\n ").split() for x in f]
vocab = PairVocab(vocab_lines, cuda=torch.cuda.is_available())

# 2. Setup Args
args = argparse.Namespace(
    vocab=vocab,
    atom_vocab=common_atom_vocab,
    rnn_type='LSTM',
    hidden_size=250,
    embed_size=250,
    latent_size=32,
    depthT=15,
    depthG=15,
    diterT=1,
    diterG=3,
    dropout=0.0
)

# 3. Load Model
model = HierVAE(args).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully!")

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    else:
        return obj

# --- EVALUATION ---
results = []
with open(DATA_PATH, 'r') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

total_mols = len(lines)
print(f"Processing {total_mols} molecules...")

for i, smi_in in enumerate(lines):
    try:
        mol_batch = MolGraph.tensorize([smi_in], vocab, common_atom_vocab)
        mol_batch = move_to_device(mol_batch, DEVICE)
        
        with torch.no_grad():
            smi_out_list = model.reconstruct(mol_batch)
            smi_out = smi_out_list[0] if smi_out_list else None

        row = {
            'id': i + 1,
            'smiles_in': smi_in,
            'smiles_out': smi_out if smi_out else "",
            'valid': 0,
            'exact': 0,
            'tanimoto': np.nan,
            'delta_atoms': np.nan,
            'delta_bonds': np.nan
        }

        if smi_out:
            m_in = Chem.MolFromSmiles(smi_in)
            m_out = Chem.MolFromSmiles(smi_out)
            
            if m_in and m_out:
                can_in = Chem.MolToSmiles(m_in, isomericSmiles=True)
                can_out = Chem.MolToSmiles(m_out, isomericSmiles=True)
                
                row['valid'] = 1
                row['exact'] = 1 if can_in == can_out else 0
                row['tanimoto'] = DataStructs.TanimotoSimilarity(
                    AllChem.GetMorganFingerprintAsBitVect(m_in, 2, nBits=2048),
                    AllChem.GetMorganFingerprintAsBitVect(m_out, 2, nBits=2048)
                )
                row['delta_atoms'] = abs(m_in.GetNumAtoms() - m_out.GetNumAtoms())
                row['delta_bonds'] = abs(m_in.GetNumBonds() - m_out.GetNumBonds())

        results.append(row)
        if (i + 1) % 50 == 0 or (i + 1) == total_mols:
            print(f"  Progress: {i+1}/{total_mols}")

    except Exception:
        continue

# --- SAVE & STATS ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

valid_df = df[df['valid'] == 1].copy()
validity_rate = (len(valid_df) / total_mols) * 100
exact_match_rate = (valid_df['exact'].sum() / total_mols) * 100
avg_tanimoto = valid_df['tanimoto'].mean() if not valid_df.empty else 0.0

# --- HISTOGRAM GENERATION ---
if not valid_df.empty:
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_df['tanimoto'], bins=20, kde=True, color='teal')
    plt.title('Molecular Reconstruction: Tanimoto Similarity Distribution', fontsize=14)
    plt.xlabel('Tanimoto Similarity (Morgan Fingerprint)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(avg_tanimoto, color='red', linestyle='--', label=f'Avg: {avg_tanimoto:.3f}')
    plt.legend()
    plt.savefig(PLOT_FILE)
    print(f"Histogram saved as {PLOT_FILE}")

# --- PRINT FINAL REPORT ---
print("\n" + "="*45)
print("PART A: RECONSTRUCTION PERFORMANCE")
print("="*45)
print(f"Model Validity Rate:   {validity_rate:.2f}%")
print(f"Exact Match Accuracy: {exact_match_rate:.2f}%")
print(f"Average Tanimoto:      {avg_tanimoto:.4f}")
if not valid_df.empty:
    print(f"Avg Atom Delta:        {valid_df['delta_atoms'].mean():.2f}")
    print(f"Avg Bond Delta:        {valid_df['delta_bonds'].mean():.2f}")
print("-" * 45)

# --- 10 QUALITATIVE EXAMPLES ---
if not valid_df.empty:
    print("\n🔍 10 EXAMPLES FOR QUALITATIVE ANALYSIS")
    best = valid_df.nlargest(5, 'tanimoto')
    mid = valid_df.iloc[len(valid_df)//2 : len(valid_df)//2 + 5]
    examples = pd.concat([best, mid])
    
    for idx, row in examples.iterrows():
        status = "EXACT" if row['exact'] == 1 else "APPROX"
        print(f"[{status}] Tanimoto: {row['tanimoto']:.3f}")
        print(f"  IN:  {row['smiles_in']}")
        print(f"  OUT: {row['smiles_out']}\n")

print(f"📂 Full data saved to: {OUTPUT_FILE}")