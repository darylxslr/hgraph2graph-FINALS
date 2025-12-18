Antibiotic Generation & Hierarchical Graph Analysis

This repository implements a Hierarchical Graph Variational Autoencoder (HierVAE) for the generation and analysis of antibiotic-like molecules. It provides a complete pipeline including data collection, preprocessing, model training, and multi-stage evaluation** to study chemical information flow through the model.

🛠 Environment Setup

It is recommended to use a Conda environment (annaconda) to ensure all dependencies, such as RDKit and PyTorch, are correctly installed.

1. Create and activate environment

bash
conda create -n hgraph python=3.7
conda activate hgraph

2. Install core dependencies
   
RDKit (required for chemical sanitization and fingerprints):

bash
conda install -c rdkit rdkit
PyTorch (adjust cudatoolkit version according to your GPU):

bash
conda install pytorch cudatoolkit=11.3 -c pytorch
Supporting libraries:

bash
pip install scikit-learn pandas matplotlib networkx tqdm

Workflow Summary
- Data Collection & Cleaning
- Patent Verification
- Scrapes chemical entries from ChEMBL and verifies them against known patents.

bash
python check_chembl_patents.py
Note: Scrapes 400 entries from chembl/all.txt.

Sanitization
Cleans SMILES data to ensure chemical validity.

bash
python sanitize_data.py

Vocabulary & Preprocessing
Extract Vocabulary
Generate a hierarchical substructure vocabulary from cleaned molecules.

bash
python get_vocab.py < data/antibiotics/all_clean.txt > data/antibiotics/vocab.txt

Tensorization
Convert molecules into graph and tree tensors for model input.

bash
python preprocess.py --train data/antibiotics/all_clean.txt --vocab data/antibiotics/vocab.txt --ncpu 4 --mode single

Organize Processed Data

bash
mkdir -p data/antibiotics/processed
mv tensors-0.pkl data/antibiotics/processed/

Model Training
Setup Checkpoint Directories

bash
mkdir -p ckpt
mkdir -p ckpt/antibiotic
Train HierVAE Generator

bash
python train_generator.py \
    --train data/antibiotics/processed/ \
    --vocab data/antibiotics/vocab.txt \
    --save_dir ckpt/antibiotic-model \
    --save_iter 20 \
    --epoch 50

Evaluation & Analysis
Global Metrics & Checkpoint Analysis
Evaluate reconstruction performance across checkpoints.

bash
python evaluate_reconstruction_to_checkpoints.py

Hidden-State Probing & Partial Decoding Analysis
Analyze latent space representations and visualize structural convergence during motif-by-motif assembly.

bash
python evaluate_full_diagnostics.py

Outputs
Reconstruction metrics: results.csv

Tanimoto similarity histogram: tanimoto_histogram_visualization.png

Partial decoding growth plot: partial_decoding_plot.png

Latent space probing results: printed to terminal

These outputs provide quantitative and qualitative insight into the model’s performance and interpretability.

📁 Folder Structure
Copy code
hgraph2graph-FINALS/
│
├── ckpt/                  # Trained model checkpoints
├── data/                  # Raw and processed molecule datasets
├── hgraph/                # Core HierVAE implementation
├── evaluate_reconstructions.py  # Evaluate molecular reconstruction performance
├── evaluate_full_diagnostics.py # Full diagnostics including hidden-state probing
├── show_molecules.py      # Visualize molecules side-by-side
├── partial_decoding_plot.png    # Example plot from diagnostics
└── README.md              # This file

