import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from hgraph.mol_graph import MolGraph
from hgraph.vocab import PairVocab, common_atom_vocab
from hgraph.hgnn import HierVAE

# --- CONFIG ---
CKPT_DIR = 'ckpt/antibiotic-model'
VOCAB_PATH = 'data/antibiotics/vocab.txt'
DATA_PATH = 'data/antibiotics/all_clean.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load Vocab & Test Data
with open(VOCAB_PATH) as f:
    vocab_lines = [x.strip("\r\n ").split() for x in f]
vocab = PairVocab(vocab_lines, cuda=torch.cuda.is_available())

with open(DATA_PATH, 'r') as f:
    test_smiles = [l.strip() for l in f.readlines() if l.strip()]

# 2. Find Checkpoints
# This assumes files are named model.ckpt.100, model.ckpt.200, etc.
ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.startswith('model.ckpt')]
# Sort by the number at the end
ckpt_files.sort(key=lambda x: int(x.split('.')[-1]))

history = []

print(f"📈 Found {len(ckpt_files)} checkpoints. Starting evaluation...")

for ckpt_name in ckpt_files:
    step = int(ckpt_name.split('.')[-1])
    print(f"Evaluating Step {step}...", end=" ", flush=True)
    
    # Load Model for this specific step
    args = argparse.Namespace(
        vocab=vocab, atom_vocab=common_atom_vocab, rnn_type='LSTM',
        hidden_size=250, embed_size=250, latent_size=32,
        depthT=15, depthG=15, diterT=1, diterG=3, dropout=0.0
    )
    model = HierVAE(args).to(DEVICE)
    checkpoint = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location=DEVICE)
    state_dict = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    exact_matches = 0
    tanimoto_scores = []
    
    for smi_in in test_smiles:
        try:
            # Simple tensorize and move to device
            batch = MolGraph.tensorize([smi_in], vocab, common_atom_vocab)
            if torch.cuda.is_available():
                batch = [x.cuda() if torch.is_tensor(x) else x for x in batch]
                # Handle nested lists/tuples in batch if necessary
            
            with torch.no_grad():
                out = model.reconstruct(batch)
                smi_out = out[0] if out else None
            
            if smi_out:
                m_in = Chem.MolFromSmiles(smi_in)
                m_out = Chem.MolFromSmiles(smi_out)
                if m_in and m_out:
                    # Check Exact Match
                    if Chem.MolToSmiles(m_in) == Chem.MolToSmiles(m_out):
                        exact_matches += 1
                    # Check Tanimoto
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(m_in, 2, 2048)
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(m_out, 2, 2048)
                    tanimoto_scores.append(DataStructs.TanimotoSimilarity(fp1, fp2))
        except: continue

    history.append({
        'step': step,
        'exact_acc': (exact_matches / len(test_smiles)) * 100,
        'mean_tanimoto': np.mean(tanimoto_scores) if tanimoto_scores else 0
    })
    print(f"Acc: {history[-1]['exact_acc']:.1f}% | Tanimoto: {history[-1]['mean_tanimoto']:.3f}")

# --- 3. PLOTTING ---
df_hist = pd.DataFrame(history)

plt.figure(figsize=(12, 5))

# Plot 1: Exact Match
plt.subplot(1, 2, 1)
plt.plot(df_hist['step'], df_hist['exact_acc'], marker='o', color='blue')
plt.title('Exact Match Accuracy vs. Training Step')
plt.xlabel('Step')
plt.ylabel('Accuracy (%)')
plt.grid(True)

# Plot 2: Tanimoto
plt.subplot(1, 2, 2)
plt.plot(df_hist['step'], df_hist['mean_tanimoto'], marker='s', color='green')
plt.title('Mean Tanimoto Similarity vs. Training Step')
plt.xlabel('Step')
plt.ylabel('Mean Tanimoto')
plt.grid(True)

plt.tight_layout()
plt.savefig('checkpoint_dynamics.png')
print("\n✅ Part B Plots saved to checkpoint_dynamics.png")