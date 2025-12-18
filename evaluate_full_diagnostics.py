import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import argparse
import os

from hgraph import HierVAE, common_atom_vocab, PairVocab, MolGraph


# ==============================================================================
# CONFIGURATION — PART C
# ==============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'ckpt/antibiotic-model/model.ckpt.440'
VOCAB_PATH = 'data/antibiotics/vocab.txt'
DATA_PATH = 'data/antibiotics/all_clean.txt'

print(f"Initializing diagnostics on {DEVICE}...")


# ==============================================================================
# SHARED INITIALIZATION
# ==============================================================================
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found.")
    exit()

with open(VOCAB_PATH) as f:
    vocab_lines = [x.strip("\r\n ").split() for x in f]

vocab = PairVocab(vocab_lines, cuda=torch.cuda.is_available())

# ---------------- FIX: device added ----------------
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
    dropout=0.0,
    device=DEVICE   # ✅ REQUIRED
)

model = HierVAE(args).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint)
model.eval()

with open(DATA_PATH) as f:
    all_smiles = [l.strip() for l in f.readlines()]


# ==============================================================================
# EXPERIMENT 1: HIDDEN-STATE PROBING
# ==============================================================================
print("\nRunning Experiment 1: Hidden-State Probing...")

probe_smiles = all_smiles[:300]
probe_data = {
    "Latent_Z": [],
    "Tree_Hidden": [],
    "Graph_Hidden": []
}
targets = {
    "MW": [],
    "Rings": []
}

for smi in probe_smiles:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        _, all_tensors, _ = MolGraph.tensorize([smi], vocab, common_atom_vocab)

        tree_tensors = [
            t.to(DEVICE) if torch.is_tensor(t) else t
            for t in all_tensors[0]
        ]
        graph_tensors = [
            t.to(DEVICE) if torch.is_tensor(t) else t
            for t in all_tensors[1]
        ]

        with torch.no_grad():
            hroot, hnode, hinter, hatom = model.encoder(tree_tensors, graph_tensors)

            # (1) Global latent
            z = hroot.cpu().numpy().flatten()

            # (2) Tree decoder init
            h_tree_func = getattr(
                model.decoder,
                'h_tree_init',
                getattr(model.decoder, 'W_tree', None)
            )
            h_tree = h_tree_func(hroot) if h_tree_func else hnode.mean(dim=0)

            # (3) Graph decoder init
            h_mol_func = getattr(
                model.decoder,
                'h_mol_init',
                getattr(model.decoder, 'W_mol', None)
            )
            h_mol = h_mol_func(hroot) if h_mol_func else hatom.mean(dim=0)

        probe_data["Latent_Z"].append(z)
        probe_data["Tree_Hidden"].append(h_tree.cpu().numpy().flatten())
        probe_data["Graph_Hidden"].append(h_mol.cpu().numpy().flatten())

        targets["MW"].append(Descriptors.MolWt(mol))
        targets["Rings"].append(rdMolDescriptors.CalcNumRings(mol))

    except Exception:
        continue


# ---------------- LINEAR PROBING ----------------
probing_results = []

for stage, vectors in probe_data.items():
    X = np.array(vectors)

    mw_model = LinearRegression().fit(X, targets["MW"])
    ring_model = LinearRegression().fit(X, targets["Rings"])

    r2_mw = r2_score(targets["MW"], mw_model.predict(X))
    r2_rings = r2_score(targets["Rings"], ring_model.predict(X))

    probing_results.append({
        "Stage": stage,
        "MW (R2)": round(r2_mw, 4),
        "Ring Count (R2)": round(r2_rings, 4)
    })

print("\n" + "=" * 70)
print("PROBING RESULTS")
print("=" * 70)
print(pd.DataFrame(probing_results).to_string(index=False))


# ==============================================================================
# EXPERIMENT 2: STRUCTURAL CONVERGENCE (PARTIAL DECODING)
# ==============================================================================
print("\nRunning Experiment 2: Structural Convergence...")


def analyze_partial_growth(smiles, max_steps=12):
    try:
        mol_obj = MolGraph(smiles)
        target_mol = Chem.MolFromSmiles(smiles)

        target_fp = AllChem.GetMorganFingerprintAsBitVect(
            target_mol, 2, nBits=2048
        )

        clusters = mol_obj.clusters
        similarities = []

        for k in range(1, min(len(clusters), max_steps) + 1):
            active_atoms = sorted({a for c in clusters[:k] for a in c})

            partial = Chem.EditableMol(Chem.Mol())
            index_map = {}

            for atom_idx in active_atoms:
                atom = target_mol.GetAtomWithIdx(atom_idx)
                index_map[atom_idx] = partial.AddAtom(atom)

            for bond in target_mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a in index_map and b in index_map:
                    partial.AddBond(
                        index_map[a],
                        index_map[b],
                        bond.GetBondType()
                    )

            p_mol = partial.GetMol()

            try:
                Chem.SanitizeMol(p_mol)
                p_fp = AllChem.GetMorganFingerprintAsBitVect(
                    p_mol, 2, nBits=2048
                )
                sim = DataStructs.TanimotoSimilarity(target_fp, p_fp)
            except Exception:
                sim = 0.0

            similarities.append(sim)

        return similarities

    except Exception:
        return None


# ---------------- PLOT ----------------
plt.figure(figsize=(10, 6))

plot_smiles = all_smiles[50:65]

for i, s in enumerate(plot_smiles):
    growth = analyze_partial_growth(s)
    if growth:
        plt.plot(
            range(1, len(growth) + 1),
            growth,
            marker='o',
            alpha=0.7,
            label=f"Mol {i + 1}"
        )

plt.xlabel("Decoding Step (Motif Additions)")
plt.ylabel("Tanimoto Similarity to Target")
plt.title("Experiment 2: Structural Convergence (Partial Decoding)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("partial_decoding_plot.png")


# ==============================================================================
# FINAL REPORT
# ==============================================================================
print("\n" + "=" * 70)
print("ALL DIAGNOSTICS COMPLETE")
print("1. Probing results printed above.")
print("2. Growth plot saved as 'partial_decoding_plot.png'.")
print("=" * 70)
