import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from hgraph.mol_graph import MolGraph
from hgraph.vocab import PairVocab, common_atom_vocab
from hgraph.hgnn import HierVAE


# ---------------- DEVICE SAFETY ----------------
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
    def cuda_fake(self, *args, **kwargs): return self
    torch.Tensor.cuda = cuda_fake
    torch.nn.Module.cuda = cuda_fake


# ---------------- PATHS ----------------
MODEL_PATH = 'ckpt/antibiotic-model/model.ckpt.640'
VOCAB_PATH = 'data/antibiotics/vocab.txt'
DATA_PATH = 'data/antibiotics/all_clean.txt'

CSV_OUTPUT = 'checkpoint_results.csv'
PNG_OUTPUT = 'checkpoint_evaluations.png'


# ---------------- LOAD VOCAB ----------------
with open(VOCAB_PATH, 'r') as f:
    vocab_lines = [x.strip("\r\n ").split() for x in f]
vocab = PairVocab(vocab_lines, cuda=(DEVICE == 'cuda'))


# ---------------- MODEL ARGS (MUST MATCH TRAINING) ----------------
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
    device=DEVICE
)

# ---------------- LOAD MODEL ----------------
model = HierVAE(args).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, tuple):
    state_dict = checkpoint[0]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("Checkpoint loaded successfully.")


# ---------------- HELPERS ----------------
def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    return obj


# ---------------- LOAD DATA ----------------
with open(DATA_PATH, 'r') as f:
    smiles_list = [x.strip() for x in f if x.strip()]

TOTAL = len(smiles_list)
print(f"Evaluating {TOTAL} molecules...")


# ---------------- EVALUATION ----------------
results = []

for i, smi_in in enumerate(smiles_list):
    try:
        mol_batch = MolGraph.tensorize([smi_in], vocab, common_atom_vocab)
        mol_batch = move_to_device(mol_batch, DEVICE)

        with torch.no_grad():
            smi_out = model.reconstruct(mol_batch)[0]

        row = {
            "id": i + 1,
            "smiles_in": smi_in,
            "smiles_out": smi_out,
            "valid": 0,
            "exact": 0,
            "tanimoto": np.nan,
            "delta_atoms": np.nan,
            "delta_bonds": np.nan
        }

        m_in = Chem.MolFromSmiles(smi_in)
        m_out = Chem.MolFromSmiles(smi_out)

        if m_in and m_out:
            row["valid"] = 1
            can_in = Chem.MolToSmiles(m_in, isomericSmiles=True)
            can_out = Chem.MolToSmiles(m_out, isomericSmiles=True)
            row["exact"] = int(can_in == can_out)

            fp_in = AllChem.GetMorganFingerprintAsBitVect(m_in, 2, nBits=2048)
            fp_out = AllChem.GetMorganFingerprintAsBitVect(m_out, 2, nBits=2048)
            row["tanimoto"] = DataStructs.TanimotoSimilarity(fp_in, fp_out)

            row["delta_atoms"] = abs(m_in.GetNumAtoms() - m_out.GetNumAtoms())
            row["delta_bonds"] = abs(m_in.GetNumBonds() - m_out.GetNumBonds())

        results.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == TOTAL:
            print(f"Progress: {i+1}/{TOTAL}")

    except Exception:
        continue


# ---------------- SAVE CSV ----------------
df = pd.DataFrame(results)
df.to_csv(CSV_OUTPUT, index=False)


# ---------------- METRICS ----------------
valid_df = df[df["valid"] == 1]

validity_rate = len(valid_df) / TOTAL * 100
exact_rate = valid_df["exact"].sum() / TOTAL * 100
avg_tanimoto = valid_df["tanimoto"].mean()


# ---------------- ONE VISUAL PNG ----------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Validity
axes[0].bar(["Checkpoint"], [validity_rate])
axes[0].set_ylim(0, 100)
axes[0].set_title("Validity Rate (%)")

# Exact match
axes[1].bar(["Checkpoint"], [exact_rate])
axes[1].set_ylim(0, 100)
axes[1].set_title("Exact Match Accuracy (%)")

# Tanimoto distribution
sns.histplot(valid_df["tanimoto"], bins=20, kde=True, ax=axes[2])
axes[2].axvline(avg_tanimoto, color="red", linestyle="--",
                label=f"Avg = {avg_tanimoto:.3f}")
axes[2].set_xlim(0, 1)
axes[2].set_title("Tanimoto Similarity Distribution")
axes[2].legend()

plt.suptitle("HierVAE Checkpoint Evaluation", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(PNG_OUTPUT, dpi=300)
plt.close()


# ---------------- FINAL REPORT ----------------
print("\n" + "=" * 50)
print("CHECKPOINT EVALUATION SUMMARY")
print("=" * 50)
print(f"Validity Rate:        {validity_rate:.2f}%")
print(f"Exact Match Accuracy:{exact_rate:.2f}%")
print(f"Average Tanimoto:     {avg_tanimoto:.4f}")
print(f"\nCSV saved to: {CSV_OUTPUT}")
print(f"PNG saved to: {PNG_OUTPUT}")
