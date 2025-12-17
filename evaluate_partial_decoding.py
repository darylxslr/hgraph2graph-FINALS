import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse

from hgraph import HierVAE, common_atom_vocab, PairVocab, MolGraph

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'ckpt/antibiotic-model/model.ckpt.440'
VOCAB_PATH = 'data/antibiotics/vocab.txt'
DATA_PATH = 'data/antibiotics/all_clean.txt'

# --- INITIALIZATION ---
with open(VOCAB_PATH) as f:
    vocab_lines = [x.strip("\r\n ").split() for x in f]
vocab = PairVocab(vocab_lines, cuda=torch.cuda.is_available())

args = argparse.Namespace(
    vocab=vocab, atom_vocab=common_atom_vocab, rnn_type='LSTM',
    hidden_size=250, embed_size=250, latent_size=32,
    depthT=15, depthG=15, diterT=1, diterG=3, dropout=0.0
)

model = HierVAE(args).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint)
model.eval()

with open(DATA_PATH) as f:
    smiles_list = [l.strip() for l in f.readlines()[:300]]

probe_data = {"Latent_Z": [], "Tree_Hidden": [], "Graph_Hidden": []}
targets = {"MW": [], "Rings": []}

print(f"🧪 Running Hidden-State Probing on {DEVICE}...")

for i, smi in enumerate(smiles_list):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        
        _, all_tensors, _ = MolGraph.tensorize([smi], vocab, common_atom_vocab)
        tree_tensors = [t.to(DEVICE) if torch.is_tensor(t) else t for t in all_tensors[0]]
        graph_tensors = [t.to(DEVICE) if torch.is_tensor(t) else t for t in all_tensors[1]]
        
        with torch.no_grad():
            # Probing 3 distinct points in the hierarchy
            hroot, hnode, hinter, hatom = model.encoder(tree_tensors, graph_tensors)
            
            # Point 1: Global Latent Space
            z = hroot.cpu().numpy().flatten()
            
            # Point 2: Tree Decoder Initial State (Mid-level motifs)
            h_tree_func = getattr(model.decoder, 'h_tree_init', getattr(model.decoder, 'W_tree', None))
            h_tree = h_tree_func(hroot) if h_tree_func else hnode.mean(dim=0)
            
            # Point 3: Graph Decoder Initial State (Fine-grained assembly)
            h_mol_func = getattr(model.decoder, 'h_mol_init', getattr(model.decoder, 'W_mol', None))
            h_mol = h_mol_func(hroot) if h_mol_func else hatom.mean(dim=0)

            probe_data["Latent_Z"].append(z)
            probe_data["Tree_Hidden"].append(h_tree.cpu().numpy().flatten())
            probe_data["Graph_Hidden"].append(h_mol.cpu().numpy().flatten())
            
            targets["MW"].append(Descriptors.MolWt(mol))
            targets["Rings"].append(rdMolDescriptors.CalcNumRings(mol))

    except Exception:
        continue

# --- EVALUATION ---
results = []
for stage, vectors in probe_data.items():
    X = np.array(vectors)
    r2_mw = r2_score(targets["MW"], LinearRegression().fit(X, targets["MW"]).predict(X))
    r2_rings = r2_score(targets["Rings"], LinearRegression().fit(X, targets["Rings"]).predict(X))
    results.append({"Stage": stage, "MW (R2)": round(r2_mw, 4), "Ring Count (R2)": round(r2_rings, 4)})

print("\n" + "="*70 + "\n EXPERIMENT 1: HIDDEN-STATE PROBING RESULTS\n" + "="*70)
print(pd.DataFrame(results).to_string(index=False))
print("="*70)