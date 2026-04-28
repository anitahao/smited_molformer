"""
Mechanistic Interpretability of SMI-TED — Experiment 1b.

Variant of smi_ted_exp1.py that correlates proxy attention against
graph-topological distance (shortest bond-path length) instead of 3D
Euclidean distance.  Motivation: SMI-TED sees SMILES as a 1-D sequence
and may attend to bond-graph structure rather than 3D geometry.

Run from Desktop/smi_ted/:
    python smi_ted_exp1b.py [path/to/smiles.csv]
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

BASE_DIR   = Path(__file__).parent.resolve()
MODEL_DIR  = BASE_DIR / 'inference' / 'smi_ted_light'
sys.path.insert(0, str(BASE_DIR / 'inference' / 'smi_ted_light'))

RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures' / 'exp1b'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem

SEED   = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Atom-only subset of load.py's PATTERN; [H] excluded so cur counts heavy atoms only
ATOM_PATTERN = re.compile(r'\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p')


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
        print("No CSV supplied")
        smiles = []
    random.shuffle(smiles)
    return smiles[:n_samples]


# ─────────────────────────────────────────────────────────────────────
# Token-to-atom mapping
# ─────────────────────────────────────────────────────────────────────

def get_atom_map(smi, tokenizer):
    """Map each token to a heavy-atom index (or -1 for non-atom tokens)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None

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

    full = [-1] + atom_map + [-1]   # wrap for <bos> and <eos>
    return full, mol, smi


def atom_token_indices(full_map):
    return [i for i, a in enumerate(full_map) if a >= 0]


# ─────────────────────────────────────────────────────────────────────
# Graph distance (bond hops between heavy atoms)
# ─────────────────────────────────────────────────────────────────────

def get_graph_distance_matrix(mol):
    """Return shortest-path bond-hop distance matrix over heavy atoms only.

    Uses RDKit's Floyd-Warshall implementation; never fails for a valid mol.
    Returns float array of shape (n_heavy, n_heavy).
    """
    mol_no_h = Chem.RemoveHs(mol)
    dist = Chem.GetDistanceMatrix(mol_no_h)   # (n_heavy, n_heavy), integer hops
    return dist.astype(float)


# ─────────────────────────────────────────────────────────────────────
# QKV → proxy attention
# ─────────────────────────────────────────────────────────────────────

def qk_to_proxy_attention(queries, keys):
    """Proxy softmax attention from captured Q, K (post-rotary).

    queries / keys : (N, L, H, d_k)
    returns        : (H, L, L)  numpy
    """
    q = queries[0].float()   # (L, H, d_k)
    k = keys[0].float()
    d_k   = q.shape[-1]
    scale = d_k ** 0.5
    q = q.permute(1, 0, 2)  # (H, L, d_k)
    k = k.permute(1, 0, 2)
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    attn   = torch.softmax(scores, dim=-1)
    return attn.numpy()      # (H, L, L)


# ─────────────────────────────────────────────────────────────────────
# Experiment 1b: Attention × Graph-Distance Correlation
# ─────────────────────────────────────────────────────────────────────

def extract_attention_and_distances(model, smiles_list, n_layers, n_heads,
                                    max_molecules=300):
    from fast_transformers.events import EventDispatcher, QKVEvent

    tokenizer  = model.tokenizer
    enc        = model.encoder
    dispatcher = EventDispatcher.get('')

    attn_modules = {
        enc.blocks.layers[i].attention: i for i in range(n_layers)
    }
    qkv_store = {}

    def listener(event):
        if isinstance(event, QKVEvent) and event.source in attn_modules:
            layer_idx = attn_modules[event.source]
            qkv_store[layer_idx] = (
                event.queries.detach().cpu(),
                event.keys.detach().cpu(),
            )

    handle = dispatcher.listen(QKVEvent, listener)
    results = []

    for smi in tqdm(smiles_list, desc='Exp1b — attention extraction'):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() < 3:
            continue

        dist_mat = get_graph_distance_matrix(mol)

        full_map, mol_obj, raw_smi = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue
        a_idx = atom_token_indices(full_map)
        if len(a_idx) != dist_mat.shape[0]:
            continue

        idx, mask = model.tokenize(raw_smi)
        qkv_store.clear()

        with torch.no_grad():
            enc(idx, mask)

        if len(qkv_store) == 0:
            continue

        n_atoms = dist_mat.shape[0]
        atom_atts = []
        for layer in range(n_layers):
            if layer not in qkv_store:
                atom_atts.append(np.zeros((n_heads, n_atoms, n_atoms)))
                continue
            q, k = qkv_store[layer]
            proxy = qk_to_proxy_attention(q, k)          # (H, L, L)
            sub   = proxy[:, a_idx, :][:, :, a_idx]      # (H, n_atoms, n_atoms)
            atom_atts.append(sub)

        atom_atts = np.array(atom_atts)   # (n_layers, n_heads, n_atoms, n_atoms)

        results.append({
            'smiles':          smi,
            'dist_matrix':     dist_mat,
            'atom_attentions': atom_atts,
            'num_atoms':       n_atoms,
        })

        if len(results) >= max_molecules:
            break

    dispatcher.remove(handle)
    print(f'Processed {len(results)} molecules for Exp 1b')
    return results


def compute_correlations(results, n_layers, n_heads):
    arrays = {k: np.zeros((n_layers, n_heads)) for k in
              ['cosine', 'pearson', 'spearman', 'short', 'medium', 'long']}
    counts = np.zeros((n_layers, n_heads))

    for res in tqdm(results, desc='Exp1b — correlations'):
        dist = res['dist_matrix']
        att  = res['atom_attentions']
        n    = res['num_atoms']
        if n < 3:
            continue

        # Use inverse bond distance so closer atoms → higher expected attention
        inv_dist = np.zeros_like(dist)
        inv_dist[dist > 0] = 1.0 / dist[dist > 0]

        idx      = np.triu_indices(n, k=1)
        d_flat   = dist[idx]
        inv_flat = inv_dist[idx]
        if len(d_flat) < 3:
            continue

        # Bond-hop strata: neighbors (1), 2-3 hops, 4+ hops
        sm  = d_flat == 1
        med = (d_flat >= 2) & (d_flat <= 3)
        lg  = d_flat >= 4

        for layer in range(min(n_layers, att.shape[0])):
            for head in range(min(n_heads, att.shape[1])):
                a_flat = att[layer, head, :n, :n][idx]
                if a_flat.std() < 1e-10 or inv_flat.std() < 1e-10:
                    continue
                try:
                    arrays['cosine'][layer, head]   += 1 - cosine_dist(a_flat, inv_flat)
                    arrays['pearson'][layer, head]   += pearsonr(a_flat, inv_flat)[0]
                    arrays['spearman'][layer, head]  += spearmanr(a_flat, inv_flat)[0]
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


def plot_exp1b(corr, n_layers, n_heads):
    lt  = list(range(1, n_layers + 1))
    ht  = list(range(1, n_heads + 1))
    ann = n_layers <= 12 and n_heads <= 12

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, n_layers * 0.5 + 1)))
    for ax, key, title in zip(axes,
            ['cosine', 'pearson', 'spearman'],
            ['Cosine Similarity', 'Pearson r', 'Spearman ρ']):
        sns.heatmap(corr[key], ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=ht, yticklabels=lt,
                    annot=ann, fmt='.2f', annot_kws={'size': 7})
        ax.set_xlabel('Head'); ax.set_ylabel('Layer')
        ax.set_title(f'{title}\n(proxy attention vs 1/bond-hops)')
    plt.suptitle('SMI-TED — Attention × Graph Distance', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smited_exp1b_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, n_layers * 0.5 + 1)))
    for ax, key, title in zip(axes,
            ['short', 'medium', 'long'],
            ['Neighbors (1 hop)', '2–3 hops', '4+ hops']):
        sns.heatmap(corr[key], ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=ht, yticklabels=lt,
                    annot=ann, fmt='.2f', annot_kws={'size': 7})
        ax.set_xlabel('Head'); ax.set_ylabel('Layer')
        ax.set_title(f'Spearman ρ — {title}')
    plt.suptitle('SMI-TED — Bond-Hop Stratified Correlation', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smited_exp1b_stratified.png', dpi=150, bbox_inches='tight')
    plt.close()

    sp = corr['spearman']
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = np.arange(1, n_layers + 1)
    ax.errorbar(layers, sp.mean(1), yerr=sp.std(1), marker='o', capsize=3, lw=2)
    ax.set_xlabel('Layer', fontsize=12); ax.set_ylabel('Mean Spearman ρ', fontsize=12)
    ax.set_title('SMI-TED — Layer-wise Attention × Graph Distance\n(mean ± std across heads)')
    ax.set_xticks(layers); ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'smited_exp1b_layerwise.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Experiment 1b plots saved.')


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    csv_path = "./data/qm9.csv"

    print('=' * 65)
    print('Mechanistic Interpretability — SMI-TED  (Exp 1b: Graph Dist)')
    print('=' * 65)
    print(f'Timestamp : {datetime.now().isoformat()}')

    model, n_layers, n_heads = load_model()
    smiles = load_smiles(csv_path, n_samples=500)

    print('\n' + '=' * 65)
    print('EXPERIMENT 1b: Proxy Attention × Graph-Topological Distance')
    print('=' * 65)

    exp1b_raw = extract_attention_and_distances(
        model, smiles, n_layers, n_heads, max_molecules=300)
    corr = compute_correlations(exp1b_raw, n_layers, n_heads)
    plot_exp1b(corr, n_layers, n_heads)

    save_corr = {k: v.tolist() for k, v in corr.items()}
    with open(RESULTS_DIR / 'smited_exp1b_correlations.json', 'w') as f:
        json.dump(save_corr, f, indent=2)

    sp   = corr['spearman']
    best = np.unravel_index(sp.argmax(), sp.shape)
    print(f'\nMean Spearman ρ : {sp.mean():.4f} (±{sp.std():.4f})')
    print(f'Best head       : Layer {best[0]+1}, Head {best[1]+1}  (ρ={sp[best]:.4f})')
    print(f'Layer-wise means: {["%.3f" % x for x in sp.mean(1)]}')


if __name__ == '__main__':
    main()
