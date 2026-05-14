"""
Mechanistic Interpretability of SMI-TED's SMILES Pretraining.

Experiments:
1. Attention-distance correlation (proxy softmax attention from QKV events,
   since SMI-TED uses linear attention that never materializes an n×n matrix)
2. Linear probing for chemical properties at each encoder layer

Run from Desktop/smi_ted/:
    python smited_interp.py [path/to/smiles.csv]

If no CSV is given, a small built-in molecule set is used for testing.
The CSV must have a column named 'smiles'.
"""

import os
import sys
import re
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime

# ── add inference folder so load.py is importable ─────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
MODEL_DIR  = BASE_DIR / 'inference' / 'smi_ted_light'
sys.path.insert(0, str(BASE_DIR / 'inference' / 'smi_ted_light'))

RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures' / 'exp2'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import RidgeCV                                                                              
from sklearn.metrics import r2_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV, LogisticRegressionCV
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem

SEED   = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Atom-only subset of load.py's PATTERN — identifies atom tokens within a tokenized sequence
ATOM_PATTERN = re.compile(r'\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

def load_model():
    from load import load_smi_ted
    print(f"Loading SMI-TED from {MODEL_DIR}")
    model = load_smi_ted(
        folder=str(MODEL_DIR),
        ckpt_filename='smi-ted-Light_40.pt',
        vocab_filename='bert_vocab_curated.txt',
    )
    if DEVICE == 'cuda':
        model.encoder.cuda()
    model.encoder.eval()
    cfg = model.config
    n_layers = cfg['n_layer']
    n_heads  = cfg['n_head']
    print(f"Architecture: {n_layers} layers × {n_heads} heads  |  device: {DEVICE}")
    return model, n_layers, n_heads


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_smiles(csv_path=None, n_samples=500):
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        col = 'smiles' if 'smiles' in df.columns else df.columns[0]
        smiles = df[col].dropna().tolist()
        print(f"Loaded {len(smiles)} SMILES from {csv_path}")
    else:
        print(f"No CSV supplied")

    random.shuffle(smiles)
    return smiles[:n_samples]


# ─────────────────────────────────────────────────────────────────────
# Token-to-atom mapping
# ─────────────────────────────────────────────────────────────────────

def get_atom_map(smiles, tokenizer):
    """Map each non-special token to an atom index (or -1).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    # Same call as MolTranBertTokenizer._tokenize()
    tokens = tokenizer.regex_tokenizer.findall(smiles)
    atom_map = []
    cur = 0
    n   = mol.GetNumAtoms()

    for tok in tokens:
        if cur >= n:
            atom_map.append(-1)
            continue
        if ATOM_PATTERN.fullmatch(tok):
            atom_map.append(cur); cur += 1
        else:
            atom_map.append(-1)

    # wrap with -1 for <bos> and <eos>
    full = [-1] + atom_map + [-1]
    return full, mol, smiles


def atom_token_indices(full_map):
    return [i for i, a in enumerate(full_map) if a >= 0]


# ─────────────────────────────────────────────────────────────────────
# RDKit helpers
# ─────────────────────────────────────────────────────────────────────

def get_atom_properties(mol):
    return [{
        'atom_type':  a.GetSymbol(),
        'hybridization': a.GetHybridization(),
        'is_aromatic': a.GetIsAromatic(),
        'is_in_ring':  a.IsInRing(),
        'chiral_tag': a.GetChiralTag(),

        'degree':      a.GetDegree(),
        'formal_charge': a.GetFormalCharge(),
        'total_valence': a.GetTotalValence()
    } for a in mol.GetAtoms()]


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Linear Probing
# ─────────────────────────────────────────────────────────────────────

def extract_hidden_states(model, smiles_list, n_layers, max_molecules=500):
    tokenizer = model.tokenizer
    enc       = model.encoder

    layer_out  = {i: [] for i in range(n_layers + 1)}  # 0 = embedding
    atom_labels = []
    captured   = {}
    hooks      = []

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h.detach().cpu()
        return fn

    # Embedding hook
    hooks.append(enc.tok_emb.register_forward_hook(make_hook(0)))

    # Encoder layer hooks
    for i, layer in enumerate(enc.blocks.layers[:n_layers]):
        hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    processed = 0
    for smi in tqdm(smiles_list, desc='Exp2 — hidden states'):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        full_map, mol_obj, raw_smi = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue
        a_idx = atom_token_indices(full_map)
        if len(a_idx) != mol.GetNumAtoms():
            continue

        idx, mask = model.tokenize(raw_smi)
        idx = idx.to(DEVICE)
        mask = mask.to(DEVICE)

        captured.clear()

        with torch.no_grad():
            enc(idx, mask)

        for layer_idx, hs in captured.items():
            # hs: (1, seq_len, d_model) — batch size 1
            atom_hs = hs[0][a_idx].numpy()   # (n_atoms, d_model)
            layer_out[layer_idx].append(atom_hs)

        for prop in get_atom_properties(mol_obj):
            atom_labels.append({**prop, 'mol_id': processed})

        processed += 1
        if processed >= max_molecules:
            break

    for h in hooks:
        h.remove()

    layer_emb = {
        i: np.concatenate(arrs, axis=0)
        for i, arrs in layer_out.items() if arrs
    }
    print(f'Extracted hidden states: {processed} molecules, {len(atom_labels)} atoms')
    return layer_emb, atom_labels

def train_test(atom_labels):
    # Molecule-level split — keep all atoms of a molecule on the same side
    df = pd.DataFrame(atom_labels)
    mol_ids = df['mol_id'].values
    unique_mols = np.unique(mol_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_mols)
   
    split_mol = int(0.8 * len(unique_mols))
    train_mols = set(unique_mols[:split_mol])
    tr = np.where(np.isin(mol_ids, list(train_mols)))[0]
    te = np.where(~np.isin(mol_ids, list(train_mols)))[0]

    return tr, te

def run_linear_probing(layer_emb, atom_labels, tr, te):
    df = pd.DataFrame(atom_labels)
    tasks = {
        'atom_type': df['atom_type'].values,
        'hybridization': df['hybridization'].astype(str).values,
        'is_aromatic': df['is_aromatic'].astype(int).values,
        'is_in_ring': df['is_in_ring'].astype(int).values,
        'chiral_tag': df['chiral_tag'].astype(str).values,

        'degree': df['degree'].values,
        'formal_charge': df['formal_charge'].values,
        'total_valence': df['total_valence'].values
    }

    results = {}
    for task, labels in tasks.items():
        print(f'\nProbing: {task}')
        le = LabelEncoder()
        y  = le.fit_transform(labels)
        print(f'  Classes: {le.classes_}')
        task_res = {}
        for layer_idx in sorted(layer_emb.keys()):
            X = layer_emb[layer_idx]
            if len(X) != len(y):
                continue
            reg = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1, C=1.0))
            reg.fit(X[tr], y[tr])                                                                                                 
            pred = reg.predict(X[te])   
            acc  = accuracy_score(y[te], pred)
            f1   = f1_score(y[te], pred, average='weighted')
            task_res[layer_idx] = {'accuracy': acc, 'f1': f1}
            print(f'  Layer {layer_idx}: Acc={acc:.4f}  F1={f1:.4f}')
        results[task] = task_res

    return results

def run_dummy_classifier(atom_labels, tr, te):
    df = pd.DataFrame(atom_labels)
    tasks = {
        'atom_type': df['atom_type'].values,
        'hybridization': df['hybridization'].astype(str).values,
        'is_aromatic': df['is_aromatic'].astype(int).values,
        'is_in_ring': df['is_in_ring'].astype(int).values,
        'chiral_tag': df['chiral_tag'].astype(str).values,

        'degree': df['degree'].values,
        'formal_charge': df['formal_charge'].values,
        'total_valence': df['total_valence'].values
    }

    results = {}
    for task, label in tasks.items():
        print(f'\nDummyClassifier: {task}')
        le = LabelEncoder()                                                                                           
        y  = le.fit_transform(label)
        dummy = DummyClassifier(strategy='most_frequent')                                                             
        dummy.fit(np.zeros((len(tr), 1)), y[tr])
        acc = accuracy_score(y[te], dummy.predict(np.zeros((len(te), 1))))
        results[task] = {'accuracy': acc}                                            
        print(f'{task}: baseline acc = {acc:.4f}')    

    return results   


def plot_exp2(probing_results, dummy_results, n_layers):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for task, res in probing_results.items():                                                                         
        layers = sorted(res.keys())                                                                                   
        line, = axes[0].plot(layers, [res[l]['accuracy'] for l in layers], marker='o', label=task, lw=2)              
        axes[1].plot(layers, [res[l]['f1'] for l in layers], marker='o', label=task, lw=2, color=line.get_color())    
        baseline = dummy_results[task]['accuracy']                                                                    
        axes[0].axhline(baseline, ls='--', color=line.get_color(), alpha=0.5)     
        axes[1].axhline(baseline, ls='--', color=line.get_color(), alpha=0.5)                                     
    for ax, ylabel, title in zip(axes, ['Accuracy', 'F1 (weighted)'],                                                 
                                ['Probe Accuracy by Layer', 'Probe F1 by Layer']):
        ax.set_xlabel('Layer', fontsize=12); ax.set_ylabel(ylabel, fontsize=12)                                       
        ax.set_title(f'SMI-TED — {title}', fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)                                                                         
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smited_exp2_probing.png', dpi=400, bbox_inches='tight')                                
    plt.close()           

    print('Experiment 2 plots saved.')


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    csv_path = "./data/qm9.csv"

    print('=' * 65)
    print('Mechanistic Interpretability — SMI-TED')
    print('=' * 65)
    print(f'Timestamp : {datetime.now().isoformat()}')

    model, n_layers, n_heads = load_model()
    smiles = load_smiles(csv_path, n_samples=1000)

    # ── Experiment 2 ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('EXPERIMENT 2: Linear Probing for Chemical Properties')
    print('=' * 65)

    layer_emb, atom_labels = extract_hidden_states(
        model, smiles, n_layers, max_molecules=1000)
    tr, te = train_test(atom_labels)
    probe_res = run_linear_probing(layer_emb, atom_labels, tr, te)
    dummy_res = run_dummy_classifier(atom_labels, tr, te)
    plot_exp2(probe_res, dummy_res, n_layers)

    with open(RESULTS_DIR / 'smited_exp2_probing.json', 'w') as f:
        json.dump({task: {str(k): v for k, v in res.items()}
                   for task, res in probe_res.items()}, f, indent=2)

    print('\n' + '=' * 65)
    print('DONE')
    print(f'Results : {RESULTS_DIR}')
    print(f'Figures : {FIGURES_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
