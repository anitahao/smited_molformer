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
FIGURES_DIR = BASE_DIR / 'figures' / 'exp1a'
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

def get_atom_map(smi, tokenizer):
    """Map each non-special token to an atom index (or -1).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None

    # Same call as MolTranBertTokenizer._tokenize()
    tokens = tokenizer.regex_tokenizer.findall(smi)
    atom_map = []
    cur = 0
    n = mol.GetNumHeavyAtoms()

    for tok in tokens:
        if cur >= n:
            atom_map.append(-1)
            continue
        if ATOM_PATTERN.fullmatch(tok) and tok != '[H]':
            atom_map.append(cur); cur += 1
        else:
            atom_map.append(-1)

    # wrap with -1 for <bos> and <eos>
    full = [-1] + atom_map + [-1]
    return full, mol, smi


def atom_token_indices(full_map):
    return [i for i, a in enumerate(full_map) if a >= 0]


# ─────────────────────────────────────────────────────────────────────
# RDKit helpers
# ─────────────────────────────────────────────────────────────────────

def get_3d_distance_matrix(mol):
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, p) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    conf  = mol.GetConformer()
    heavy = [i for i in range(mol.GetNumAtoms())
             if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    pos   = np.array([list(conf.GetAtomPosition(i)) for i in heavy])
    diff  = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# ─────────────────────────────────────────────────────────────────────
# QKV → proxy attention  (Exp 1 core)
# ─────────────────────────────────────────────────────────────────────

def qk_to_proxy_attention(queries, keys):
    """Compute proxy softmax attention from Q and K tensors.

    SMI-TED uses linear attention (no explicit n×n matrix).
    We reconstruct a proxy by computing softmax(Q K^T / sqrt(d_k)).

    queries / keys : (N, L, H, d_k)  — straight from QKVEvent
    returns        : (H, L, L)  numpy, batch dim removed (N=1)
    """
    q = queries[0].float()   # (L, H, d_k)
    k = keys[0].float()       # (L, H, d_k)
    d_k   = q.shape[-1]
    scale = d_k ** 0.5

    # (H, L, d_k) @ (H, d_k, L) -> (H, L, L)
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    attn   = torch.softmax(scores, dim=-1)
    return attn.numpy()      # (H, L, L)


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Attention-Distance Correlation
# ─────────────────────────────────────────────────────────────────────

def extract_attention_and_distances(model, smiles_list, n_layers, n_heads,
                                    max_molecules=10):
    from fast_transformers.events import EventDispatcher, QKVEvent

    tokenizer  = model.tokenizer
    enc        = model.encoder
    dispatcher = EventDispatcher.get('')

    # Map each RotateAttentionLayer → layer index
    attn_modules = {
        enc.blocks.layers[i].attention: i for i in range(n_layers)
    }
    qkv_store = {}

    def listener(event):
        if isinstance(event, QKVEvent) and event.source in attn_modules:
            idx = attn_modules[event.source]
            qkv_store[idx] = (
                event.queries.detach().cpu(),
                event.keys.detach().cpu(),
            )

    handle = dispatcher.listen(QKVEvent, listener)
    results = []

    for smi in tqdm(smiles_list, desc='Exp1 — attention extraction'):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 3:
            continue

        dist_mat = get_3d_distance_matrix(mol)
        if dist_mat is None:
            continue

        full_map, mol_obj, canonical_smi = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue
        a_idx = atom_token_indices(full_map)
        if len(a_idx) != dist_mat.shape[0]:
            continue

        # Tokenize canonical SMILES — matches what get_atom_map tokenized
        idx, mask = model.tokenize(canonical_smi)
        qkv_store.clear()

        with torch.no_grad():
            enc(idx, mask)

        if len(qkv_store) == 0:
            continue

        # Use dist_mat row count — heavy atoms only, excludes explicit H
        n_atoms = dist_mat.shape[0]
        atom_atts = []
        for layer in range(n_layers):
            if layer not in qkv_store:
                atom_atts.append(np.zeros((n_heads, n_atoms, n_atoms)))
                continue
            q, k = qkv_store[layer]
            proxy = qk_to_proxy_attention(q, k)   # (H, L, L)
            # slice to atom positions
            sub = proxy[:, :, :]
            sub = sub[:, a_idx, :][:, :, a_idx]   # (H, n_atoms, n_atoms)
            atom_atts.append(sub)

        atom_atts = np.array(atom_atts)  # (n_layers, n_heads, n_atoms, n_atoms)

        results.append({
            'smiles':         smi,
            'dist_matrix':    dist_mat[:n_atoms, :n_atoms],
            'atom_attentions': atom_atts,
            'num_atoms':      n_atoms,
        })

        if len(results) >= max_molecules:
            break

    dispatcher.remove(handle)
    print(f'Processed {len(results)} molecules for Exp 1')
    return results


def compute_correlations(results, n_layers, n_heads):
    arrays = {k: np.zeros((n_layers, n_heads)) for k in
              ['cosine', 'pearson', 'spearman', 'short', 'medium', 'long']}
    counts = np.zeros((n_layers, n_heads))

    for res in tqdm(results, desc='Exp1 — correlations'):
        dist = res['dist_matrix']
        att  = res['atom_attentions']
        n    = res['num_atoms']
        if n < 3:
            continue

        inv_dist = np.zeros_like(dist)
        inv_dist[dist > 0] = 1.0 / dist[dist > 0]

        idx       = np.triu_indices(n, k=1)
        d_flat    = dist[idx]
        inv_flat  = inv_dist[idx]
        if len(d_flat) < 3:
            continue

        sm  = d_flat <= 2.0
        med = (d_flat > 2.0) & (d_flat <= 4.0)
        lg  = d_flat > 4.0

        for layer in range(min(n_layers, att.shape[0])):
            for head in range(min(n_heads, att.shape[1])):
                a_flat = att[layer, head, :n, :n][idx]
                if a_flat.std() < 1e-10 or inv_flat.std() < 1e-10:
                    continue
                try:
                    arrays['cosine'][layer, head]  += 1 - cosine_dist(a_flat, inv_flat)
                    arrays['pearson'][layer, head]  += pearsonr(a_flat, inv_flat)[0]
                    arrays['spearman'][layer, head] += spearmanr(a_flat, inv_flat)[0]
                    counts[layer, head] += 1
                except Exception:
                    continue
                for mask_arr, key in [(sm, 'short'), (med, 'medium'), (lg, 'long')]:
                    if mask_arr.sum() >= 3:
                        try:
                            arrays[key][layer, head] += spearmanr(
                                a_flat[mask_arr], inv_flat[mask_arr])[0]
                        except Exception:
                            pass

    valid = counts > 0
    for arr in arrays.values():
        arr[valid] /= counts[valid]
    arrays['counts'] = counts
    return arrays


def plot_exp1(corr, n_layers, n_heads):
    lt = list(range(1, n_layers + 1))
    ht = list(range(1, n_heads + 1))
    ann = n_layers <= 12 and n_heads <= 12

    # Main heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, n_layers * 0.5 + 1)))
    for ax, key, title in zip(axes,
            ['cosine', 'pearson', 'spearman'],
            ['Cosine Similarity', 'Pearson r', 'Spearman ρ']):
        sns.heatmap(corr[key], ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=ht, yticklabels=lt,
                    annot=ann, fmt='.2f', annot_kws={'size': 7})
        ax.set_xlabel('Head'); ax.set_ylabel('Layer')
        ax.set_title(f'{title}\n(proxy attention vs 1/distance)')
    plt.suptitle('SMI-TED — Attention × 3D Distance', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smited_exp1a_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Stratified
    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, n_layers * 0.5 + 1)))
    for ax, key, title in zip(axes,
            ['short', 'medium', 'long'],
            ['Short (≤2 Å)', 'Medium (2–4 Å)', 'Long (>4 Å)']):
        sns.heatmap(corr[key], ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=ht, yticklabels=lt,
                    annot=ann, fmt='.2f', annot_kws={'size': 7})
        ax.set_xlabel('Head'); ax.set_ylabel('Layer')
        ax.set_title(f'Spearman ρ — {title}')
    plt.suptitle('SMI-TED — Distance-Stratified Correlation', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smited_exp1a_stratified.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Layer-wise mean ± std
    sp = corr['spearman']
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = np.arange(1, n_layers + 1)
    ax.errorbar(layers, sp.mean(1), yerr=sp.std(1), marker='o', capsize=3, lw=2)
    ax.set_xlabel('Layer', fontsize=12); ax.set_ylabel('Mean Spearman ρ', fontsize=12)
    ax.set_title('SMI-TED — Layer-wise Attention-Distance Correlation\n(mean ± std across heads)')
    ax.set_xticks(layers); ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smited_exp1a_layerwise.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Experiment 1 plots saved.')


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
    smiles = load_smiles(csv_path, n_samples=500)

    # ── Experiment 1 ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('EXPERIMENT 1: Proxy Attention × 3D Distance Correlation')
    print('=' * 65)

    exp1_raw = extract_attention_and_distances(
        model, smiles, n_layers, n_heads, max_molecules=300)
    corr = compute_correlations(exp1_raw, n_layers, n_heads)
    plot_exp1(corr, n_layers, n_heads)

    save_corr = {k: v.tolist() for k, v in corr.items()}
    with open(RESULTS_DIR / 'smited_exp1a_correlations.json', 'w') as f:
        json.dump(save_corr, f, indent=2)

    sp   = corr['spearman']
    best = np.unravel_index(sp.argmax(), sp.shape)
    print(f'\nMean Spearman ρ : {sp.mean():.4f} (±{sp.std():.4f})')
    print(f'Best head       : Layer {best[0]+1}, Head {best[1]+1}  (ρ={sp[best]:.4f})')
    print(f'Layer-wise means: {["%.3f" % x for x in sp.mean(1)]}')


if __name__ == '__main__':
    main()
