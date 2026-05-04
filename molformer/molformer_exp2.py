"""
Mechanistic Interpretability of MolFormer — Experiment 2.

Linear probing for chemical properties at each encoder layer.

Changes from previous version:
  1. Use output_hidden_states=True (official HuggingFace API)
     instead of forward hooks
  2. Add StandardScaler before LogisticRegression
  3. Both frequency baseline and random-input baseline
  4. 8 atom properties
  5. QM9 as main experiment + ESOL as validation
  6. One figure per property, unified y-axis (0-1)
  7. Molecule-level 80/20 train/test split (no data leakage)

Run from smited_molformer/:
    python molformer/molformer_exp2.py
"""

import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SEED   = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR    = Path(__file__).parent.parent.resolve()
DATA_DIR    = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results' / 'molformer'
FIGURES_DIR = RESULTS_DIR / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

QM9_PATH  = DATA_DIR / 'qm9.csv'
ESOL_PATH = DATA_DIR / 'esol.csv'


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ─────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModel, AutoTokenizer
    print("Loading MolFormer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(
        'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Architecture: {n_layers} layers | device: {DEVICE}")
    return model, tokenizer, n_layers


# ─────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────

def load_smiles(csv_path, n_samples=1000, max_len=150):
    df  = pd.read_csv(csv_path)
    col = 'smiles' if 'smiles' in df.columns else df.columns[-1]
    smiles = df[col].dropna().tolist()
    smiles = [s for s in smiles if len(s) <= max_len]
    random.shuffle(smiles)
    print(f"Loaded {len(smiles[:n_samples])} SMILES from {csv_path.name}")
    return smiles[:n_samples]


# ─────────────────────────────────────────────────────────────────────
# Token-to-Atom Mapping
# ─────────────────────────────────────────────────────────────────────

def get_atom_map(smi, tokenizer):
    """Map each token to an atom index (-1 for non-atom tokens)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None

    tokens = tokenizer.tokenize(smi)
    atom_map = []
    cur = 0
    n   = mol.GetNumAtoms()

    for tok in tokens:
        if cur >= n:
            atom_map.append(-1)
            continue
        if tok.startswith('['):
            atom_map.append(cur); cur += 1
        elif len(tok) == 1 and tok in 'BCNOPSFIcnops':
            atom_map.append(cur); cur += 1
        elif tok in ('Cl', 'Br', 'Si', 'Se', 'se'):
            atom_map.append(cur); cur += 1
        else:
            atom_map.append(-1)

    # wrap with -1 for <bos> and <eos>
    full = [-1] + atom_map + [-1]
    return full, mol


def atom_token_indices(full_map):
    return [i for i, a in enumerate(full_map) if a >= 0]


# ─────────────────────────────────────────────────────────────────────
# Atom Properties — 8 properties
# ─────────────────────────────────────────────────────────────────────

def get_atom_properties(mol):
    return [{
        'atom_type':     atom.GetSymbol(),
        'hybridization': str(atom.GetHybridization()),
        'is_aromatic':   atom.GetIsAromatic(),
        'is_in_ring':    atom.IsInRing(),
        'chiral_tag':    str(atom.GetChiralTag()),
        'degree':        atom.GetDegree(),
        'formal_charge': atom.GetFormalCharge(),
        'total_valence': atom.GetTotalValence(),
    } for atom in mol.GetAtoms()]


# ─────────────────────────────────────────────────────────────────────
# Hidden State Extraction
# Uses output_hidden_states=True (official HuggingFace API)
# No hooks needed — more reliable than manual hook injection
# ─────────────────────────────────────────────────────────────────────

def extract_hidden_states(model, tokenizer, smiles_list,
                           n_layers, max_molecules=1000,
                           use_random_input=False):
    """
    Extract hidden states from all 13 layers at atom token positions.

    If use_random_input=True, replace the embedding output with
    random Gaussian noise before passing through encoder layers.
    This serves as a random-input activation baseline.

    Uses output_hidden_states=True instead of hooks (Anita's approach),
    which is the official HuggingFace API and more reliable.
    For random input, a single hook on the embedding layer is still
    needed to inject noise before the encoder layers process it.
    """
    layer_out   = {i: [] for i in range(n_layers + 1)}
    atom_labels = []
    hooks       = []

    # Random-input baseline: replace embedding output with noise
    if use_random_input and hasattr(model, 'embeddings'):
        def randomize_hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.randn_like(output[0]),) + output[1:]
            return torch.randn_like(output)
        hooks.append(
            model.embeddings.register_forward_hook(randomize_hook))

    processed = 0
    desc = "Extracting hidden states" + (" (random)" if use_random_input else "")

    for smi in tqdm(smiles_list, desc=desc):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        full_map, mol_obj = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue

        a_idx = atom_token_indices(full_map)
        if len(a_idx) != mol.GetNumAtoms():
            continue

        inputs = tokenizer(smi, return_tensors='pt', padding=False,
                           truncation=True, max_length=202)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        seq_len     = inputs['input_ids'].shape[1]
        a_idx_valid = [i for i in a_idx if i < seq_len]
        if len(a_idx_valid) != mol.GetNumAtoms():
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of (1, seq_len, d_model)
        # [0] = embedding, [1..n_layers] = transformer layers
        for layer_idx, hs in enumerate(outputs.hidden_states):
            atom_hs = hs[0][a_idx_valid].cpu().numpy()
            layer_out[layer_idx].append(atom_hs)

        for prop in get_atom_properties(mol_obj):
            prop['mol_id'] = processed
            atom_labels.append(prop)

        processed += 1
        if processed >= max_molecules:
            break

    for h in hooks:
        h.remove()

    layer_emb = {
        i: np.concatenate(arrs, axis=0)
        for i, arrs in layer_out.items() if arrs
    }
    print(f"Extracted hidden states: {processed} molecules, "
          f"{len(atom_labels)} atoms"
          + (" [random input]" if use_random_input else ""))
    return layer_emb, atom_labels


# ─────────────────────────────────────────────────────────────────────
# Train/Test Split
# ─────────────────────────────────────────────────────────────────────

def train_test_split(atom_labels):
    """Molecule-level 80/20 split to prevent data leakage."""
    df      = pd.DataFrame(atom_labels)
    mol_ids = df['mol_id'].values
    unique_mols = np.unique(mol_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_mols)

    split     = int(0.8 * len(unique_mols))
    train_set = set(unique_mols[:split])
    tr = np.where( np.isin(mol_ids, list(train_set)))[0]
    te = np.where(~np.isin(mol_ids, list(train_set)))[0]
    return tr, te


# ─────────────────────────────────────────────────────────────────────
# Linear Probing
# Uses StandardScaler + LogisticRegression pipeline
# ─────────────────────────────────────────────────────────────────────

def run_linear_probing(layer_emb, atom_labels, tr, te,
                        layer_emb_random=None):
    """
    Train linear probes for 8 chemical properties.

    Uses make_pipeline(StandardScaler(), LogisticRegression(...))
    to normalize features before classification — critical for
    reliable results with high-dimensional hidden states.

    Records:
      - accuracy / f1         : model activations
      - frequency_baseline    : DummyClassifier(most_frequent)
      - accuracy_random       : probe on random-input activations
    """
    df = pd.DataFrame(atom_labels)

    probing_tasks = {
        'atom_type':     df['atom_type'].values,
        'hybridization': df['hybridization'].astype(str).values,
        'is_aromatic':   df['is_aromatic'].astype(int).values,
        'is_in_ring':    df['is_in_ring'].astype(int).values,
        'chiral_tag':    df['chiral_tag'].astype(str).values,
        'degree':        df['degree'].values,
        'formal_charge': df['formal_charge'].values,
        'total_valence': df['total_valence'].values,
    }

    results = {}

    for task, labels in probing_tasks.items():
        print(f"\nProbing: {task}")
        le = LabelEncoder()
        y  = le.fit_transform(labels)
        print(f"  Classes: {le.classes_}")

        y_train, y_test = y[tr], y[te]

        if len(np.unique(y_train)) < 2:
            print("  Skipped (only one class in training set)")
            continue

        # Frequency baseline
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(np.zeros((len(tr), 1)), y_train)
        freq_acc = accuracy_score(
            y_test, dummy.predict(np.zeros((len(te), 1))))
        print(f"  Frequency baseline: {freq_acc:.4f}")

        task_res = {}

        for layer_idx in sorted(layer_emb.keys()):
            X = layer_emb[layer_idx]
            if len(X) != len(y):
                continue

            # StandardScaler + LogisticRegression pipeline
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=SEED,
                                   n_jobs=-1, C=1.0))
            clf.fit(X[tr], y_train)
            pred = clf.predict(X[te])

            acc = accuracy_score(y_test, pred)
            f1  = f1_score(y_test, pred, average='weighted')
            entry = {
                'accuracy':           acc,
                'f1':                 f1,
                'frequency_baseline': freq_acc,
            }

            # Random-input baseline
            if layer_emb_random is not None:
                X_r = layer_emb_random.get(layer_idx)
                if X_r is not None and len(X_r) == len(y):
                    clf_r = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=1000,
                                           random_state=SEED,
                                           n_jobs=-1, C=1.0))
                    clf_r.fit(X_r[tr], y_train)
                    entry['accuracy_random'] = accuracy_score(
                        y_test, clf_r.predict(X_r[te]))

            task_res[layer_idx] = entry

            rand_str = (f", RandomAcc={entry['accuracy_random']:.4f}"
                        if 'accuracy_random' in entry else "")
            print(f"  Layer {layer_idx}: "
                  f"Acc={acc:.4f}, F1={f1:.4f}{rand_str}")

        results[task] = task_res

    return results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_results(probing_results, title_prefix, fig_dir):
    """
    One figure per property. Each figure shows:
      - Model activations     (solid blue)
      - Random-input baseline (dashed gray)
      - Frequency baseline    (dotted red horizontal line)
    Unified y-axis (0 to 1) for cross-property comparison.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    for task, task_res in probing_results.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        layers = sorted(task_res.keys())
        accs   = [task_res[l]['accuracy'] for l in layers]
        ax.plot(layers, accs, marker='o', linewidth=2,
                color='steelblue', label='Model activations')

        if 'accuracy_random' in task_res[layers[0]]:
            accs_r = [task_res[l]['accuracy_random'] for l in layers]
            ax.plot(layers, accs_r, marker='s', linewidth=2,
                    linestyle='--', color='gray',
                    label='Random input baseline')

        freq_acc = task_res[layers[0]]['frequency_baseline']
        ax.axhline(freq_acc, linestyle=':', linewidth=2,
                   color='tomato',
                   label=f'Frequency baseline ({freq_acc:.2f})')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{title_prefix} Linear Probe: {task}', fontsize=13)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / f'exp2_probing_{task}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run_experiment(model, tokenizer, n_layers, smiles_list,
                   dataset_name, max_molecules):
    """Run full probing pipeline for one dataset."""
    print(f"\n{'='*65}")
    print(f"Dataset: {dataset_name}  ({len(smiles_list)} SMILES)")
    print('='*65)

    # Normal activations
    layer_emb, atom_labels = extract_hidden_states(
        model, tokenizer, smiles_list, n_layers,
        max_molecules=max_molecules)

    # Random-input baseline
    layer_emb_random, _ = extract_hidden_states(
        model, tokenizer, smiles_list, n_layers,
        max_molecules=max_molecules, use_random_input=True)

    tr, te = train_test_split(atom_labels)

    probing_results = run_linear_probing(
        layer_emb, atom_labels, tr, te, layer_emb_random)

    # Save figures
    fig_dir = FIGURES_DIR / dataset_name.lower()
    plot_results(probing_results, f'MOLFormer ({dataset_name})', fig_dir)

    # Save JSON
    out_json = RESULTS_DIR / f'exp2_probing_{dataset_name.lower()}.json'
    saveable = {
        task: {str(k): v for k, v in res.items()}
        for task, res in probing_results.items()
    }
    with open(out_json, 'w') as f:
        json.dump(saveable, f, indent=2)
    print(f"Results saved to: {out_json}")

    return probing_results


def main():
    print('=' * 65)
    print('MOLFormer — Experiment 2: Linear Probing')
    print('=' * 65)
    print(f'Timestamp : {datetime.now().isoformat()}')
    print(f'Device    : {DEVICE}')
    print(f'Seed      : {SEED}')

    model, tokenizer, n_layers = load_model()

    # ── Main experiment: QM9 ──────────────────────────────────────────
    qm9_smiles = load_smiles(QM9_PATH, n_samples=1000)
    run_experiment(model, tokenizer, n_layers,
                   qm9_smiles, 'QM9', max_molecules=1000)

    # ── Validation: ESOL ─────────────────────────────────────────────
    esol_smiles = load_smiles(ESOL_PATH, n_samples=1000)
    run_experiment(model, tokenizer, n_layers,
                   esol_smiles, 'ESOL', max_molecules=len(esol_smiles))

    print('\n' + '=' * 65)
    print('EXPERIMENT 2 COMPLETE')
    print(f'Figures : {FIGURES_DIR}')
    print(f'Results : {RESULTS_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
