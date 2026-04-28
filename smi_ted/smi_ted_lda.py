"""
SMI-TED — Supervised Dimensionality Reduction and Atom Representation Analysis.

Why PCA shows no clusters despite linear probing working well:
  PCA is unsupervised and variance-maximizing — it finds the directions of
  maximum spread, which may be dominated by sequence length or molecular size.
  The chemical property signal is real but lives in low-variance directions
  that PCA never looks at.  These experiments directly target those directions.

Experiments:
  E1 : LDA projection      — supervised 2D scatter, layers x properties
  E2 : Probe directions    — project onto logistic regression weight axes
  E3 : UMAP atom embeddings — non-linear 2D at each analyzed layer
  E4 : Centroid drift      — Fisher inter/intra ratio across all layers

Run from Desktop/smi_ted/:
    python smi_ted_lda.py
"""

import sys
import re
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime

BASE_DIR  = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / 'inference' / 'smi_ted_light'
sys.path.insert(0, str(MODEL_DIR))

RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures' / 'lda'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem

SEED   = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_ATOM_SAMPLES = 300
ATOM_PATTERN   = re.compile(r'\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p')

N_LAYERS          = None
LAYERS_TO_ANALYZE = None


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ─────────────────────────────────────────────────────────────────────
# Model / data loading
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

    global N_LAYERS, LAYERS_TO_ANALYZE
    N_LAYERS = model.config['n_layer']
    n = N_LAYERS
    LAYERS_TO_ANALYZE = sorted({0, n // 4, n // 2, 3 * n // 4, n})
    print(f"Architecture: {N_LAYERS} layers x {model.config['n_head']} heads")
    print(f"Analyzing layers: {LAYERS_TO_ANALYZE}")
    return model


def load_smiles():
    path = BASE_DIR / 'data' / 'qm9.csv'
    df = pd.read_csv(path).dropna(subset=['smiles'])
    n_req = N_ATOM_SAMPLES * 4
    if len(df) > n_req:
        df = df.sample(n=n_req, random_state=SEED)
    print(f"Loaded {len(df)} SMILES from QM9")
    return df['smiles'].tolist()


# ─────────────────────────────────────────────────────────────────────
# Tokenization / atom mapping
# ─────────────────────────────────────────────────────────────────────

def get_atom_map(smi, tokenizer):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None
    tokens   = tokenizer.regex_tokenizer.findall(smi)
    atom_map = []
    cur, n   = 0, mol.GetNumHeavyAtoms()
    for tok in tokens:
        if cur >= n:
            atom_map.append(-1); continue
        if ATOM_PATTERN.fullmatch(tok) and tok != '[H]':
            atom_map.append(cur); cur += 1
        else:
            atom_map.append(-1)
    return [-1] + atom_map + [-1], mol, smi


def atom_token_indices(full_map):
    return [i for i, a in enumerate(full_map) if a >= 0]


def get_atom_properties(mol):
    mol_h = Chem.RemoveHs(mol)
    return [{
        'atom_type':     a.GetSymbol(),
        'formal_charge': a.GetFormalCharge(),
        'total_valence': a.GetTotalValence(),
        'hybridization': str(a.GetHybridization()),
        'is_aromatic':   int(a.GetIsAromatic()),
        'is_in_ring':    int(a.IsInRing()),
        'is_in_ring_5':  int(a.IsInRingSize(5)),
        'is_in_ring_6':  int(a.IsInRingSize(6)),
        'degree':        a.GetDegree(),
    } for a in mol_h.GetAtoms()]


# ─────────────────────────────────────────────────────────────────────
# Embedding extraction (all layers via forward hooks)
# ─────────────────────────────────────────────────────────────────────

def extract_atom_hidden_states(model, smiles_list):
    enc       = model.encoder
    tokenizer = model.tokenizer

    captured = {}
    hooks    = []

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h.detach().cpu()
        return fn

    hooks.append(enc.tok_emb.register_forward_hook(make_hook(0)))
    for i, layer in enumerate(enc.blocks.layers[:N_LAYERS]):
        hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    layer_out   = {i: [] for i in range(N_LAYERS + 1)}
    atom_labels = []
    processed   = 0

    for smi in tqdm(smiles_list, desc='Extracting hidden states'):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() < 2:
            continue
        full_map, mol_obj, raw_smi = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue
        a_idx = atom_token_indices(full_map)
        props = get_atom_properties(mol_obj)
        if len(a_idx) != len(props):
            continue

        idx_t, mask_t = model.tokenize(raw_smi)
        captured.clear()
        with torch.no_grad():
            enc(idx_t, mask_t)

        for layer_idx, hs in captured.items():
            layer_out[layer_idx].append(hs[0][a_idx].numpy())

        for prop in props:
            atom_labels.append({**prop, 'mol_id': processed})
        processed += 1
        if processed >= N_ATOM_SAMPLES:
            break

    for h in hooks:
        h.remove()

    layer_emb = {
        i: np.concatenate(arrs, axis=0)
        for i, arrs in layer_out.items() if arrs
    }
    print(f"Extracted: {processed} molecules, {len(atom_labels)} atoms")
    return layer_emb, atom_labels


# ─────────────────────────────────────────────────────────────────────
# Property rendering helpers
# ─────────────────────────────────────────────────────────────────────

TASK_SPECS = {
    'atom_type':     {'kind': 'categorical', 'cmap': 'tab10',   'label': 'Atom Type'},
    'hybridization': {'kind': 'categorical', 'cmap': 'Set1',    'label': 'Hybridization'},
    'degree':        {'kind': 'ordinal',     'cmap': 'viridis', 'label': 'Degree'},
    'total_valence': {'kind': 'ordinal',     'cmap': 'viridis', 'label': 'Total Valence'},
    'formal_charge': {'kind': 'ordinal',     'cmap': 'RdBu',    'label': 'Formal Charge'},
    'is_aromatic':   {'kind': 'binary',      'label': 'Aromatic'},
    'is_in_ring':    {'kind': 'binary',      'label': 'In Ring'},
    'is_in_ring_5':  {'kind': 'binary',      'label': 'Ring (5)'},
    'is_in_ring_6':  {'kind': 'binary',      'label': 'Ring (6)'},
}

PROP_GROUPS = [
    ['atom_type', 'hybridization', 'degree', 'is_aromatic', 'is_in_ring'],
    ['formal_charge', 'total_valence', 'is_in_ring_5', 'is_in_ring_6'],
]


def scatter_colored(ax, X2d, raw_labels, spec):
    kind   = spec['kind']
    labels = np.asarray(raw_labels)

    if kind == 'binary':
        for val, color, lbl in [(1, 'crimson', 'Yes'), (0, 'steelblue', 'No')]:
            m = labels.astype(int) == val
            ax.scatter(X2d[m, 0], X2d[m, 1], c=color, s=3, alpha=0.4,
                       label=lbl, rasterized=True)
        ax.legend(fontsize=6, markerscale=3)

    elif kind == 'ordinal':
        vals = labels.astype(int)
        uv   = sorted(set(vals))
        cs   = plt.cm.get_cmap(spec['cmap'])(np.linspace(0, 1, max(len(uv), 2)))
        for i, v in enumerate(uv):
            m = vals == v
            ax.scatter(X2d[m, 0], X2d[m, 1], c=[cs[i]], s=3, alpha=0.5,
                       label=str(v), rasterized=True)
        ax.legend(title=spec['label'], fontsize=5, markerscale=3)

    else:  # categorical
        str_vals = labels.astype(str)
        unique   = sorted(set(str_vals))
        cmap     = plt.cm.get_cmap(spec['cmap'])
        cs       = cmap(np.linspace(0, 1, max(len(unique), 2)))
        for i, v in enumerate(unique):
            m = str_vals == v
            ax.scatter(X2d[m, 0], X2d[m, 1], c=[cs[i % len(cs)]], s=3, alpha=0.5,
                       label=v, rasterized=True)
        ax.legend(fontsize=5, markerscale=3)


def balance_carbon(df):
    counts = df['atom_type'].value_counts()
    if 'C' not in counts.index or len(counts) <= 1:
        return list(range(len(df)))
    cap     = int(2 * counts.iloc[1])
    c_idx   = df[df['atom_type'] == 'C'].index.tolist()
    nc_idx  = df[df['atom_type'] != 'C'].index.tolist()
    sampled = np.random.choice(c_idx, min(len(c_idx), cap), replace=False).tolist()
    return sorted(sampled + nc_idx)


# ─────────────────────────────────────────────────────────────────────
# E1: LDA projection grids
# ─────────────────────────────────────────────────────────────────────

def run_lda_grids(layer_emb, df, balanced_idx):
    print("\n" + "=" * 60)
    print("E1: LDA Projection Grids")
    print("=" * 60)

    df_bal = df.iloc[balanced_idx].reset_index(drop=True)
    n_lyrs = len(LAYERS_TO_ANALYZE)

    for g_num, props in enumerate(PROP_GROUPS, 1):
        n_props = len(props)
        fig, axes = plt.subplots(n_lyrs, n_props,
                                 figsize=(4 * n_props, 3.5 * n_lyrs))
        if n_lyrs == 1:
            axes = [axes]

        for row, layer in enumerate(LAYERS_TO_ANALYZE):
            X_sc = StandardScaler().fit_transform(layer_emb[layer][balanced_idx])

            for col, prop in enumerate(props):
                ax    = axes[row][col]
                spec  = TASK_SPECS[prop]
                le    = LabelEncoder()
                y     = le.fit_transform(df_bal[prop].astype(str).values)
                n_cls = len(le.classes_)

                if n_cls < 2 or X_sc.shape[0] < n_cls + 1:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    n_comp = min(2, n_cls - 1)
                    try:
                        lda   = LinearDiscriminantAnalysis(n_components=n_comp)
                        X_lda = lda.fit_transform(X_sc, y)
                        if n_comp == 1:
                            jitter = np.random.uniform(-0.5, 0.5, len(X_lda))
                            X2d    = np.column_stack([X_lda[:, 0], jitter])
                            ax.set_xlabel('LD1', fontsize=7)
                            ax.set_ylabel('(jitter)', fontsize=7)
                        else:
                            X2d = X_lda[:, :2]
                            ax.set_xlabel('LD1', fontsize=7)
                            ax.set_ylabel('LD2', fontsize=7)
                        scatter_colored(ax, X2d, df_bal[prop].values, spec)
                    except Exception as e:
                        ax.text(0.5, 0.5, str(e)[:40], ha='center', va='center',
                                transform=ax.transAxes, fontsize=5)

                if row == 0:
                    ax.set_title(spec['label'], fontsize=11)
                if col == 0:
                    ylbl = 'LD2' if n_cls > 2 else '(jitter)'
                    ax.set_ylabel(f'Layer {layer}\n{ylbl}', fontsize=9)
                ax.tick_params(labelsize=6)

        fig.suptitle(f'SMI-TED — LDA Projection (Group {g_num})', fontsize=14, y=1.01)
        plt.tight_layout()
        out = FIGURES_DIR / f'e1_lda_grid{g_num}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────
# E2: Probe-direction projection
# Binary props  -> 1D score on x, random jitter on y.
# Multi-class   -> top-2 right singular vectors of the weight matrix.
# ─────────────────────────────────────────────────────────────────────

def run_probe_directions(layer_emb, df, balanced_idx):
    print("\n" + "=" * 60)
    print("E2: Probe-Direction Projection")
    print("=" * 60)

    df_bal = df.iloc[balanced_idx].reset_index(drop=True)
    n_lyrs = len(LAYERS_TO_ANALYZE)

    for g_num, props in enumerate(PROP_GROUPS, 1):
        n_props = len(props)
        fig, axes = plt.subplots(n_lyrs, n_props,
                                 figsize=(4 * n_props, 3.5 * n_lyrs))
        if n_lyrs == 1:
            axes = [axes]

        for row, layer in enumerate(LAYERS_TO_ANALYZE):
            X_sc = StandardScaler().fit_transform(layer_emb[layer][balanced_idx])

            for col, prop in enumerate(props):
                ax    = axes[row][col]
                spec  = TASK_SPECS[prop]
                le    = LabelEncoder()
                y     = le.fit_transform(df_bal[prop].astype(str).values)
                n_cls = len(le.classes_)

                if n_cls < 2:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    try:
                        clf = LogisticRegression(max_iter=1000, random_state=SEED,
                                                 n_jobs=-1)
                        clf.fit(X_sc, y)
                        W = clf.coef_   # (n_cls_or_1, d_model)

                        if W.shape[0] == 1:
                            scores = X_sc @ W[0]
                            jitter = np.random.uniform(-0.5, 0.5, len(scores))
                            X2d    = np.column_stack([scores, jitter])
                            ax.set_xlabel('Probe score', fontsize=7)
                            ax.set_ylabel('(jitter)', fontsize=7)
                        else:
                            # top-2 right singular vectors of the weight matrix
                            _, _, Vt = np.linalg.svd(W, full_matrices=False)
                            X2d = X_sc @ Vt[:2].T
                            ax.set_xlabel('PD1', fontsize=7)
                            ax.set_ylabel('PD2', fontsize=7)

                        scatter_colored(ax, X2d, df_bal[prop].values, spec)
                    except Exception as e:
                        ax.text(0.5, 0.5, str(e)[:40], ha='center', va='center',
                                transform=ax.transAxes, fontsize=5)

                if row == 0:
                    ax.set_title(spec['label'], fontsize=11)
                if col == 0:
                    ylbl = 'PD2' if n_cls > 2 else '(jitter)'
                    ax.set_ylabel(f'Layer {layer}\n{ylbl}', fontsize=9)
                ax.tick_params(labelsize=6)

        fig.suptitle(f'SMI-TED — Probe-Direction Projection (Group {g_num})',
                     fontsize=14, y=1.01)
        plt.tight_layout()
        out = FIGURES_DIR / f'e2_probe_directions_grid{g_num}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────
# E3: UMAP on atom embeddings
# ─────────────────────────────────────────────────────────────────────

def run_umap_atoms(layer_emb, df, balanced_idx):
    print("\n" + "=" * 60)
    print("E3: UMAP on Atom Embeddings")
    print("=" * 60)

    try:
        import umap as umap_lib
    except ImportError:
        print("  umap-learn not installed — skipping E3.  (pip install umap-learn)")
        return

    df_bal    = df.iloc[balanced_idx].reset_index(drop=True)
    all_props = [p for group in PROP_GROUPS for p in group]
    n_cols    = 5
    n_rows    = (len(all_props) + n_cols - 1) // n_cols

    for layer in LAYERS_TO_ANALYZE:
        X_sc = StandardScaler().fit_transform(layer_emb[layer][balanced_idx])
        print(f"  Running UMAP at layer {layer} ({X_sc.shape[0]} atoms)...")
        reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine',
                                random_state=SEED)
        X_umap  = reducer.fit_transform(X_sc)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = axes.flatten()

        for ax_i, prop in enumerate(all_props):
            ax   = axes_flat[ax_i]
            spec = TASK_SPECS[prop]
            scatter_colored(ax, X_umap, df_bal[prop].values, spec)
            ax.set_title(spec['label'], fontsize=10)
            ax.set_xlabel('UMAP1', fontsize=7)
            ax.set_ylabel('UMAP2', fontsize=7)
            ax.tick_params(labelsize=6)

        for ax in axes_flat[len(all_props):]:
            ax.set_visible(False)

        fig.suptitle(f'SMI-TED — Atom UMAP (Layer {layer})', fontsize=13, y=1.01)
        plt.tight_layout()
        out = FIGURES_DIR / f'e3_umap_layer{layer}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────
# E4: Centroid drift — Fisher inter/intra ratio across all layers
# Higher ratio = classes are more separated = property more linearly
# separable at that layer.
# ─────────────────────────────────────────────────────────────────────

def run_centroid_drift(layer_emb, df):
    print("\n" + "=" * 60)
    print("E4: Centroid Drift (Fisher ratio across layers)")
    print("=" * 60)

    all_layers = sorted(layer_emb.keys())
    results    = {prop: [] for prop in TASK_SPECS}

    for layer in tqdm(all_layers, desc='Centroid drift'):
        X = layer_emb[layer]
        for prop in TASK_SPECS:
            raw     = df[prop].values.astype(str)
            classes = np.unique(raw)
            if len(classes) < 2:
                results[prop].append(np.nan)
                continue

            centroids  = []
            intra_list = []
            for c in classes:
                m = raw == c
                if m.sum() < 2:
                    continue
                cx = X[m]
                centroid = cx.mean(axis=0)
                centroids.append(centroid)
                intra_list.append(np.mean(np.linalg.norm(cx - centroid, axis=1)))

            if len(centroids) < 2:
                results[prop].append(np.nan)
                continue

            inter = np.mean([
                np.linalg.norm(centroids[i] - centroids[j])
                for i in range(len(centroids))
                for j in range(i + 1, len(centroids))
            ])
            intra = np.mean(intra_list)
            results[prop].append(inter / (intra + 1e-8))

    fig, ax = plt.subplots(figsize=(10, 6))
    for prop, ratios in results.items():
        ax.plot(all_layers, ratios, marker='o', lw=2,
                label=TASK_SPECS[prop]['label'])

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Inter / Intra centroid distance (Fisher ratio)', fontsize=11)
    ax.set_title('SMI-TED — Centroid Drift Across Layers\n'
                 '(higher = more linearly separable)', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_layers)
    plt.tight_layout()
    out = FIGURES_DIR / 'e4_centroid_drift.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")

    save = {p: [None if np.isnan(v) else float(v) for v in vs]
            for p, vs in results.items()}
    with open(RESULTS_DIR / 'lda_e4_centroid_drift.json', 'w') as f:
        json.dump(save, f, indent=2)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('SMI-TED — Supervised Dimensionality Reduction')
    print('=' * 65)
    print(f'Timestamp : {datetime.now().isoformat()}')
    print(f'Device    : {DEVICE}')

    model  = load_model()
    smiles = load_smiles()

    layer_emb, atom_labels = extract_atom_hidden_states(model, smiles)

    df           = pd.DataFrame(atom_labels)
    balanced_idx = balance_carbon(df)
    print(f"Balanced atom set: {len(balanced_idx)} atoms (from {len(df)} total)")

    run_lda_grids(layer_emb, df, balanced_idx)
    run_probe_directions(layer_emb, df, balanced_idx)
    run_umap_atoms(layer_emb, df, balanced_idx)
    run_centroid_drift(layer_emb, df)

    print('\n' + '=' * 65)
    print('DONE')
    print(f'Figures : {FIGURES_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
