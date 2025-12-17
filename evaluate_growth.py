import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import argparse
import os
from hgraph import HierVAE, common_atom_vocab, PairVocab, MolGraph

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'ckpt/antibiotic-model/model.ckpt.440'
VOCAB_PATH = 'data/antibiotics/vocab.txt'
DATA_PATH = 'data/antibiotics/all_clean.txt'

with open(VOCAB_PATH) as f:
    vocab_lines = [x.strip("\r\n ").split() for x in f]
vocab = PairVocab(vocab_lines, cuda=torch.cuda.is_available())

args = argparse.Namespace(vocab=vocab, atom_vocab=common_atom_vocab, rnn_type='LSTM', 
                          hidden_size=250, embed_size=250, latent_size=32, 
                          depthT=15, depthG=15, diterT=1, diterG=3, dropout=0.0)

model = HierVAE(args).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint)
model.eval()

def analyze_partial_growth(smiles, max_steps=12):
    try:
        mol_obj = MolGraph(smiles)
        target_mol = Chem.MolFromSmiles(smiles)
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
        
        clusters = mol_obj.clusters
        similarities = []
        
        for k in range(1, min(len(clusters), max_steps) + 1):
            active_atoms = []
            for i in range(k):
                active_atoms.extend(clusters[i])
            active_atoms = sorted(list(set(active_atoms)))

            # Robust partial reconstruction
            partial_mol = Chem.EditableMol(Chem.Mol())
            old_to_new = {}
            for i, atom_idx in enumerate(active_atoms):
                atom = target_mol.GetAtomWithIdx(atom_idx)
                new_idx = partial_mol.AddAtom(atom)
                old_to_new[atom_idx] = new_idx
            
            for bond in target_mol.GetBonds():
                if bond.GetBeginAtomIdx() in old_to_new and bond.GetEndAtomIdx() in old_to_new:
                    partial_mol.AddBond(old_to_new[bond.GetBeginAtomIdx()], 
                                        old_to_new[bond.GetEndAtomIdx()], 
                                        bond.GetBondType())
            
            p_mol = partial_mol.GetMol()
            try:
                Chem.SanitizeMol(p_mol)
                p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(target_fp, p_fp)
            except:
                sim = 0.0
            similarities.append(sim)
        return similarities
    except:
        return None

print(f"🧪 Analyzing Structural Convergence...")
with open(DATA_PATH) as f:
    test_smiles = [l.strip() for l in f.readlines()[50:65]]

plt.figure(figsize=(10, 6))
for i, s in enumerate(test_smiles):
    growth = analyze_partial_growth(s)
    if growth:
        plt.plot(range(1, len(growth)+1), growth, marker='o', alpha=0.7, label=f"Mol {i+1}")

plt.xlabel("Decoding Step (Motif Additions)")
plt.ylabel("Tanimoto Similarity to Target")
plt.title("Experiment 2: Structural Convergence (Partial Decoding)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("partial_decoding_plot.png")
print("✅ Done! Plot saved.")