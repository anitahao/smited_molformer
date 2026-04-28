"""
SMI-TED PCA / UMAP Visualization of Atom Representations.

Experiments:
  3a: Atom-level PCA at layers [0,4,8,12], colored by each atom property
  3b: Atom-level UMAP at layers [0,4,8,12], colored by each atom property
  3c/3d: 3a + 3b repeated on a randomly initialized encoder (untrained baseline)

Run from Desktop/smi_ted/:
    python smi_ted_pca.py
"""

import copy
import sys
import re
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from rdkit import Chem
from rdkit.Chem import AllChem

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR  = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "inference" / "smi_ted_light"
sys.path.insert(0, str(MODEL_DIR))

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "pca"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES   = 1000    # molecules to use

ATOM_PATTERN = re.compile(r"\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p")

CATEGORICAL_PROPS = {"atom_type", "hybridization", "chiral_tag"}
BOOLEAN_PROPS     = {"is_aromatic", "is_in_ring"}
NUMERIC_PROPS     = {"degree", "formal_charge", "total_valence"}
ALL_PROPS = [
    "atom_type", "hybridization", "is_aromatic", "is_in_ring",
    "chiral_tag", "degree", "formal_charge", "total_valence",
]


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
        ckpt_filename="smi-ted-Light_40.pt",
        vocab_filename="bert_vocab_curated.txt",
    )
    model.encoder.to(DEVICE)
    model.encoder.eval()
    n_layers = model.config["n_layer"]
    print(f"Architecture: {n_layers} layers  |  device: {DEVICE}")
    return model, n_layers


def load_random_model(model):
    print("Creating randomly initialized encoder...")
    enc_rand = copy.deepcopy(model.encoder)
    for p in enc_rand.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)
    enc_rand.eval()
    return enc_rand


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_smiles(csv_path, n_samples=N_SAMPLES):
    df  = pd.read_csv(csv_path)
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    smiles = df[col].dropna().tolist()
    random.shuffle(smiles)
    print(f"Loaded {len(smiles)} SMILES (after length filter)")
    return smiles[:n_samples]


# ─────────────────────────────────────────────────────────────────────
# Token-to-atom mapping
# ─────────────────────────────────────────────────────────────────────

def get_atom_map(smi, tokenizer):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None

    tokens = tokenizer.regex_tokenizer.findall(smi)
    atom_map = []
    cur = 0
    n   = mol.GetNumHeavyAtoms()

    for tok in tokens:
        if cur >= n:
            atom_map.append(-1)
            continue
        if ATOM_PATTERN.fullmatch(tok) and tok != "[H]":
            atom_map.append(cur); cur += 1
        else:
            atom_map.append(-1)

    full = [-1] + atom_map + [-1]
    return full, mol


def atom_token_indices(full_map):
    return [i for i, a in enumerate(full_map) if a >= 0]


# ─────────────────────────────────────────────────────────────────────
# Atom properties
# ─────────────────────────────────────────────────────────────────────

def get_atom_properties(mol):
    return [{
        "atom_type":     a.GetSymbol(),
        "hybridization": str(a.GetHybridization()),
        "is_aromatic":   int(a.GetIsAromatic()),
        "is_in_ring":    int(a.IsInRing()),
        "chiral_tag":    str(a.GetChiralTag()),
        "degree":        a.GetDegree(),
        "formal_charge": a.GetFormalCharge(),
        "total_valence": a.GetTotalValence(),
    } for a in mol.GetAtoms()]


# ─────────────────────────────────────────────────────────────────────
# Hidden state extraction
# ─────────────────────────────────────────────────────────────────────

def extract_atom_embeddings(model_or_enc, tokenizer, smiles_list, layers,
                             is_encoder=False, max_molecules=500):
    """Extract per-atom hidden states at the specified layer indices.

    model_or_enc: full model (is_encoder=False) or bare encoder (is_encoder=True)
    layers: list of layer indices to capture (0 = embedding, 1..n = transformer layers)
    Returns:
        atom_emb  : dict[layer] -> np.array (n_atoms_total, d_model)
        atom_props: list of dicts, one per atom
    """
    enc = model_or_enc if is_encoder else model_or_enc.encoder

    captured = {}
    hooks    = []

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h.detach().cpu()
        return fn

    if 0 in layers:
        hooks.append(enc.tok_emb.register_forward_hook(make_hook(0)))
    for i, layer in enumerate(enc.blocks.layers):
        if (i + 1) in layers:
            hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    layer_out   = {l: [] for l in layers}
    atom_props  = []
    processed   = 0

    for smi in tqdm(smiles_list, desc="Extracting atom embeddings"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        full_map, mol_obj = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue

        a_idx = atom_token_indices(full_map)
        if len(a_idx) != mol.GetNumAtoms():
            continue

        idx_t, mask_t = (model_or_enc if not is_encoder else model_or_enc).tokenize(smi) \
            if not is_encoder else (None, None)

        if not is_encoder:
            idx_t, mask_t = model_or_enc.tokenize(smi)
            idx_t  = idx_t.to(DEVICE)
            mask_t = mask_t.to(DEVICE)
        else:
            # bare encoder — need to tokenize via parent model, handled externally
            # this branch shouldn't be reached; always pass full model
            continue

        seq_len_limit = None
        captured.clear()

        with torch.no_grad():
            enc(idx_t, mask_t)

        # check bounds
        seq_lens = [captured[l].shape[1] for l in layers if l in captured]
        if not seq_lens:
            continue
        seq_len = min(seq_lens)
        a_idx_valid = [i for i in a_idx if i < seq_len]
        if len(a_idx_valid) != mol.GetNumAtoms():
            continue

        for l in layers:
            if l not in captured:
                continue
            hs = captured[l][0][a_idx_valid].numpy()  # (n_atoms, d_model)
            layer_out[l].append(hs)

        for prop in get_atom_properties(mol_obj):
            atom_props.append(prop)

        processed += 1
        if processed >= max_molecules:
            break

    for h in hooks:
        h.remove()

    atom_emb = {
        l: np.concatenate(arrs, axis=0)
        for l, arrs in layer_out.items() if arrs
    }
    print(f"  Extracted {processed} molecules, {len(atom_props)} atoms")
    return atom_emb, atom_props


# ─────────────────────────────────────────────────────────────────────
# Random encoder wrapper — uses full model's tokenizer
# ─────────────────────────────────────────────────────────────────────

def extract_atom_embeddings_random(rand_enc, model, smiles_list, layers,
                                    max_molecules=500):
    """Same as extract_atom_embeddings but uses rand_enc for forward pass."""
    tokenizer = model.tokenizer

    captured = {}
    hooks    = []

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h.detach().cpu()
        return fn

    if 0 in layers:
        hooks.append(rand_enc.tok_emb.register_forward_hook(make_hook(0)))
    for i, layer in enumerate(rand_enc.blocks.layers):
        if (i + 1) in layers:
            hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    layer_out  = {l: [] for l in layers}
    atom_props = []
    processed  = 0

    for smi in tqdm(smiles_list, desc="Extracting random encoder embeddings"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        full_map, mol_obj = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue

        a_idx = atom_token_indices(full_map)
        if len(a_idx) != mol.GetNumAtoms():
            continue

        idx_t, mask_t = model.tokenize(smi)
        idx_t  = idx_t.to(DEVICE)
        mask_t = mask_t.to(DEVICE)

        captured.clear()
        with torch.no_grad():
            rand_enc(idx_t, mask_t)

        seq_lens = [captured[l].shape[1] for l in layers if l in captured]
        if not seq_lens:
            continue
        seq_len = min(seq_lens)
        a_idx_valid = [i for i in a_idx if i < seq_len]
        if len(a_idx_valid) != mol.GetNumAtoms():
            continue

        for l in layers:
            if l not in captured:
                continue
            hs = captured[l][0][a_idx_valid].numpy()
            layer_out[l].append(hs)

        for prop in get_atom_properties(mol_obj):
            atom_props.append(prop)

        processed += 1
        if processed >= max_molecules:
            break

    for h in hooks:
        h.remove()

    atom_emb = {
        l: np.concatenate(arrs, axis=0)
        for l, arrs in layer_out.items() if arrs
    }
    print(f"  Extracted {processed} molecules, {len(atom_props)} atoms (random enc)")
    return atom_emb, atom_props


# ─────────────────────────────────────────────────────────────────────
# Coloring helpers
# ─────────────────────────────────────────────────────────────────────

def prop_to_color_args(prop_name, values):
    """Return (c, cmap, norm, legend_handles) for scatter."""
    if prop_name in CATEGORICAL_PROPS:
        le     = LabelEncoder()
        c      = le.fit_transform(values.astype(str)).astype(float)
        n_cls  = len(le.classes_)
        cmap   = plt.cm.get_cmap("tab20", n_cls)
        norm   = plt.Normalize(0, max(n_cls - 1, 1))
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(norm(i)), markersize=6, label=str(cls))
            for i, cls in enumerate(le.classes_)
        ]
        return c, cmap, norm, handles

    elif prop_name in BOOLEAN_PROPS:
        c      = values.astype(float)
        cmap   = plt.cm.get_cmap("RdYlGn", 2)
        norm   = plt.Normalize(-0.5, 1.5)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(norm(v)), markersize=6, label=lbl)
            for v, lbl in [(0, "False"), (1, "True")]
        ]
        return c, cmap, norm, handles

    else:  # numeric
        c    = values.astype(float)
        cmap = "coolwarm" if prop_name == "formal_charge" else "viridis"
        return c, cmap, None, None


# ─────────────────────────────────────────────────────────────────────
# Experiment 3a/3c: Atom PCA grid
# ─────────────────────────────────────────────────────────────────────

def run_pca_grid(atom_emb, atom_props, layers, title_prefix, filename):
    print(f"\nAtom PCA grid ({title_prefix})")
    df = pd.DataFrame(atom_props)

    # PCA per layer
    pca_coords = {}
    for l in layers:
        if l not in atom_emb:
            continue
        X    = StandardScaler().fit_transform(atom_emb[l])
        coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
        pca_coords[l] = coords

    valid_layers = [l for l in layers if l in pca_coords]
    n_rows = len(ALL_PROPS)
    n_cols = len(valid_layers)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, prop in enumerate(ALL_PROPS):
        values = df[prop].values
        c, cmap, norm, handles = prop_to_color_args(prop, values)

        for col, layer in enumerate(valid_layers):
            ax    = axes[row, col]
            coords = pca_coords[layer]

            if norm is not None:
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=c, cmap=cmap, norm=norm, s=4, alpha=0.5)
            else:
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=c, cmap=cmap, s=4, alpha=0.5)
                plt.colorbar(sc, ax=ax, pad=0.02)

            if handles and col == n_cols - 1:
                ax.legend(handles=handles, fontsize=5, loc="upper right",
                          markerscale=1.2, framealpha=0.6)

            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)
            if col == 0:
                ax.set_ylabel(prop, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"SMI-TED — Atom PCA by Property  [{title_prefix}]",
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# ─────────────────────────────────────────────────────────────────────
# Experiment 3b/3d: Atom UMAP grid
# ─────────────────────────────────────────────────────────────────────

def run_umap_grid(atom_emb, atom_props, layers, title_prefix, filename):
    try:
        import umap
    except ImportError:
        print("umap-learn not installed — skipping UMAP experiment.")
        return

    print(f"\nAtom UMAP grid ({title_prefix})")
    df = pd.DataFrame(atom_props)

    umap_coords = {}
    for l in layers:
        if l not in atom_emb:
            continue
        X      = StandardScaler().fit_transform(atom_emb[l])
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1,
                            metric="cosine", random_state=SEED)
        umap_coords[l] = reducer.fit_transform(X)
        print(f"  UMAP done for layer {l}")

    valid_layers = [l for l in layers if l in umap_coords]
    n_rows = len(ALL_PROPS)
    n_cols = len(valid_layers)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, prop in enumerate(ALL_PROPS):
        values = df[prop].values
        c, cmap, norm, handles = prop_to_color_args(prop, values)

        for col, layer in enumerate(valid_layers):
            ax     = axes[row, col]
            coords = umap_coords[layer]

            if norm is not None:
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=c, cmap=cmap, norm=norm, s=4, alpha=0.5)
            else:
                sc = ax.scatter(coords[:, 0], coords[:, 1],
                                c=c, cmap=cmap, s=4, alpha=0.5)
                plt.colorbar(sc, ax=ax, pad=0.02)

            if handles and col == n_cols - 1:
                ax.legend(handles=handles, fontsize=5, loc="upper right",
                          markerscale=1.2, framealpha=0.6)

            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)
            if col == 0:
                ax.set_ylabel(prop, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"SMI-TED — Atom UMAP by Property  [{title_prefix}]",
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("SMI-TED — Atom PCA / UMAP Visualization")
    print("=" * 65)
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"Device    : {DEVICE}")

    model, n_layers = load_model()
    layers = [l for l in [0, 4, 8, n_layers] if l <= n_layers]
    print(f"Layers to analyze: {layers}")

    smiles = load_smiles(BASE_DIR / "data" / "hiv.csv")

    # ── Exp 3a: PCA pretrained ────────────────────────────────────────
    print("\n--- Experiment 3a: PCA (Pretrained) ---")
    atom_emb_pre, atom_props_pre = extract_atom_embeddings(
        model, model.tokenizer, smiles, layers, max_molecules=N_SAMPLES)

    run_pca_grid(atom_emb_pre, atom_props_pre, layers,
                 title_prefix="Pretrained",
                 filename="exp3a_pca_pretrained.png")

    # ── Exp 3b: UMAP pretrained ───────────────────────────────────────
    print("\n--- Experiment 3b: UMAP (Pretrained) ---")
    run_umap_grid(atom_emb_pre, atom_props_pre, layers,
                  title_prefix="Pretrained",
                  filename="exp3b_umap_pretrained.png")

    # ── Exp 3c: PCA random ────────────────────────────────────────────
    print("\n--- Experiment 3c: PCA (Random) ---")
    rand_enc = load_random_model(model)
    rand_enc.to(DEVICE)

    atom_emb_rand, atom_props_rand = extract_atom_embeddings_random(
        rand_enc, model, smiles, layers, max_molecules=N_SAMPLES)

    run_pca_grid(atom_emb_rand, atom_props_rand, layers,
                 title_prefix="Random (Untrained)",
                 filename="exp3c_pca_random.png")

    # ── Exp 3d: UMAP random ───────────────────────────────────────────
    print("\n--- Experiment 3d: UMAP (Random) ---")
    run_umap_grid(atom_emb_rand, atom_props_rand, layers,
                  title_prefix="Random (Untrained)",
                  filename="exp3d_umap_random.png")

    print("\n" + "=" * 65)
    print("DONE")
    print(f"Figures : {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
