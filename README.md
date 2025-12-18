Antibiotic Generation & Hierarchical Graph Analysis
This repository contains a specialized implementation of the Hierarchical Graph VAE (HierVAE) for the generation of antibiotic-like molecules. It includes scripts for data collection, preprocessing, model training, and a multi-stage evaluation pipeline to analyze the flow of chemical information through the model.

🛠 Environment Setup
To ensure all scripts run correctly, it is recommended to use a Conda environment. This project relies on RDKit and PyTorch, which are best managed via Conda.

1. Create and Activate Environment
Bash

conda create -n hgraph python=3.7
conda activate hgraph
2. Install Core Dependencies
Install the required scientific and chemical informatics libraries:

Bash

# Install RDKit (Required for chemical sanitization and fingerprints)
conda install -c rdkit rdkit

# Install PyTorch (Adjust 'cudatoolkit' version based on your GPU)
conda install pytorch cudatoolkit=11.3 -c pytorch

# Install supporting libraries
pip install scikit-learn pandas matplotlib networkx tqdm


Workflow Summary
1. Data Collection & Cleaning
Step 1: Patent Verification: Run the scraper to verify chemical data against known patents.

Bash
python check_chembl_patents.py
Note: Scrapes 400 entries from chembl/all.txt.

Step 2: Sanitization: Deep clean the SMILES data to ensure chemical validity.

Bash
python sanitize_data.py

2. Vocabulary & Preprocessing

Step 3: Extract Vocabulary: Extract the hierarchical substructure vocabulary from the cleaned dataset.

Bash
python get_vocab.py < data/antibiotics/all_clean.txt > data/antibiotics/vocab.txt

Step 4: Tensorization: Preprocess the molecules into graph and tree tensors.

Bash
python preprocess.py --train data/antibiotics/all_clean.txt --vocab data/antibiotics/vocab.txt --ncpu 4 --mode single

Step 5-6: Data Organization:

Bash
mkdir -p data/antibiotics/processed
mv tensors-0.pkl data/antibiotics/processed/

3. Training

Step 7: Setup Directories:

Bash
mkdir -p ckpt
mkdir -p ckpt/antibiotic

Step 8: Model Training: Train the HierVAE generator.

Bash
python train_generator.py --train data/antibiotics/processed/ --vocab data/antibiotics/vocab.txt --save_dir ckpt/antibiotic-model --save_iter 20 --epoch 50

4. Evaluation & Analysis
Step 9: Global Metrics: Measure Exact Match and Tanimoto similarity scores and Checkpoint Analysis: Review performance across different training iterations.

Bash 
python evaluate_reconstruction_to_checkpoints.py

Step 11: Hidden-State Probing (Experiment 1): Predict molecular properties from internal decoder layers. Partial Decoding Analysis (Experiment 2): Visualize structural convergence during motif-by-motif assembly.

Bash 
python evaluate_full_diagnostics.py