"""
Mechanistic Interpretability of MolFormer — Experiment 1c.

MolFormer uses standard softmax attention (not linear/FAVOR+ like SMI-TED),
so the full attention matrix is directly available via output_attentions=True.

    A_ij^h = softmax(Q_i^h · K_j^h / sqrt(d_k))

This is correlated against 3D Euclidean distance between heavy atoms.

Run from Desktop/molformer/:
    python molformer_exp1c.py
"""

import os
import sys
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime

BASE_DIR    = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "exp1c"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


def load_model():
    from transformers import AutoModel, AutoTokenizer
    print("Loading MolFormer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    print(f"Architecture: {n_layers} layers × {n_heads} heads  |  device: {DEVICE}")
    return model, tokenizer, n_layers, n_heads


def load_smiles(csv_path, n_samples=1000, max_len=150):
    df = pd.read_csv(csv_path)
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    smiles = df[col].dropna().tolist()
    smiles = [s for s in smiles if len(s) <= max_len]
    random.shuffle(smiles)
    print(f"Loaded {len(smiles)} SMILES (after length filter) from {csv_path}")
    return smiles[:n_samples]


def get_atom_map(smi, tokenizer):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None

    tokens = tokenizer.tokenize(smi)
    atom_map = []
    cur = 0
    n = mol.GetNumAtoms()

    for tok in tokens:
        if cur >= n:
            atom_map.append(-1)
            continue
        if tok.startswith("["):
            atom_map.append(cur); cur += 1
        elif len(tok) == 1 and tok in "BCNOPSFIcnops":
            atom_map.append(cur); cur += 1
        elif tok in ("Cl", "Br", "Si", "Se", "se"):
            atom_map.append(cur); cur += 1
        else:
            atom_map.append(-1)

    # wrap with -1 for BOS and EOS special tokens
    full = [-1] + atom_map + [-1]
    return full, mol


def atom_token_indices(full_map):
    return [i for i, a in enumerate(full_map) if a >= 0]


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

    conf = mol.GetConformer()
    heavy = [i for i in range(mol.GetNumAtoms())
             if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    pos  = np.array([list(conf.GetAtomPosition(i)) for i in heavy])
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def extract_attention_and_distances(model, tokenizer, smiles_list, n_layers, n_heads,
                                    max_molecules=300):
    results = []

    for smi in tqdm(smiles_list, desc="Exp1c — attention extraction"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() < 3:
            continue

        dist_mat = get_3d_distance_matrix(mol)
        if dist_mat is None:
            continue

        full_map, mol_obj = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue

        a_idx = atom_token_indices(full_map)
        if len(a_idx) != dist_mat.shape[0]:
            continue

        inputs = tokenizer(smi, return_tensors="pt", padding=False,
                           truncation=True, max_length=202)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        seq_len = inputs["input_ids"].shape[1]
        if max(a_idx) >= seq_len:
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
        n_atoms = dist_mat.shape[0]
        atom_atts = []
        for layer_att in outputs.attentions:
            # layer_att: (1, n_heads, seq_len, seq_len) → remove batch dim
            att = layer_att[0].cpu().numpy()           # (n_heads, seq_len, seq_len)
            sub = att[:, a_idx, :][:, :, a_idx]        # (n_heads, n_atoms, n_atoms)
            atom_atts.append(sub)

        atom_atts = np.array(atom_atts)   # (n_layers, n_heads, n_atoms, n_atoms)

        results.append({
            "smiles":         smi,
            "dist_matrix":    dist_mat,
            "atom_attentions": atom_atts,
            "num_atoms":      n_atoms,
        })

        if len(results) >= max_molecules:
            break

    print(f"Processed {len(results)} molecules for Exp 1c")
    return results


def compute_correlations(results, n_layers, n_heads):
    arrays = {
        k: np.zeros((n_layers, n_heads))
        for k in ["cosine", "pearson", "spearman",
                  "short_cosine",  "medium_cosine",  "long_cosine",
                  "short_pearson", "medium_pearson", "long_pearson",
                  "short_spearman","medium_spearman","long_spearman"]
    }
    counts = np.zeros((n_layers, n_heads))

    for res in tqdm(results, desc="Exp1c — correlations"):
        dist = res["dist_matrix"]
        att  = res["atom_attentions"]
        n    = res["num_atoms"]

        if n < 3:
            continue

        inv_dist = np.zeros_like(dist)
        inv_dist[dist > 0] = 1.0 / dist[dist > 0]

        idx     = np.triu_indices(n, k=1)
        d_flat  = dist[idx]
        inv_flat = inv_dist[idx]

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
                    arrays["cosine"][layer, head]   += 1 - cosine_dist(a_flat, inv_flat)
                    arrays["pearson"][layer, head]  += pearsonr(a_flat, inv_flat)[0]
                    arrays["spearman"][layer, head] += spearmanr(a_flat, inv_flat)[0]
                    counts[layer, head] += 1
                except Exception:
                    continue

                for mask_arr, key in [(sm, "short_cosine"), (med, "medium_cosine"), (lg, "long_cosine")]:
                    if mask_arr.sum() >= 3:
                        try:
                            arrays[key][layer, head] += 1 - cosine_dist(a_flat[mask_arr], inv_flat[mask_arr])
                        except Exception:
                            pass

                for mask_arr, key in [(sm, "short_pearson"), (med, "medium_pearson"), (lg, "long_pearson")]:
                    if mask_arr.sum() >= 3:
                        try:
                            arrays[key][layer, head] += pearsonr(a_flat[mask_arr], inv_flat[mask_arr])[0]
                        except Exception:
                            pass

                for mask_arr, key in [(sm, "short_spearman"), (med, "medium_spearman"), (lg, "long_spearman")]:
                    if mask_arr.sum() >= 3:
                        try:
                            arrays[key][layer, head] += spearmanr(a_flat[mask_arr], inv_flat[mask_arr])[0]
                        except Exception:
                            pass

    valid = counts > 0
    for arr in arrays.values():
        arr[valid] /= counts[valid]

    arrays["counts"] = counts
    return arrays


def plot_exp1c(corr, n_layers, n_heads):
    lt  = list(range(1, n_layers + 1))
    ht  = list(range(1, n_heads + 1))
    ann = n_layers <= 12 and n_heads <= 12

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(
        axes,
        ["cosine", "pearson", "spearman"],
        ["Cosine Similarity", "Pearson", "Spearman"],
    ):
        sns.heatmap(corr[key], ax=ax, cmap="RdBu_r", center=0,
                    xticklabels=ht, yticklabels=lt,
                    annot=ann, fmt=".2f", annot_kws={"size": 7})
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)

    plt.suptitle("MolFormer — Softmax Attention × 3D Distance", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "molformer_exp1c_heatmaps.png", dpi=400, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    for ax, key, title in zip(
        axes.flatten(),
        ["short_cosine",   "medium_cosine",   "long_cosine",
         "short_pearson",  "medium_pearson",  "long_pearson",
         "short_spearman", "medium_spearman", "long_spearman"],
        ["Short Cosine (≤2 Å)", "Medium Cosine (2–4 Å)", "Long Cosine (>4 Å)",
         "Short Pearson (≤2 Å)", "Medium Pearson (2–4 Å)", "Long Pearson (>4 Å)",
         "Short Spearman (≤2 Å)", "Medium Spearman (2–4 Å)", "Long Spearman (>4 Å)"],
    ):
        sns.heatmap(corr[key], ax=ax, cmap="RdBu_r", center=0,
                    xticklabels=ht, yticklabels=lt,
                    annot=ann, fmt=".2f", annot_kws={"size": 7})
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)

    plt.suptitle("MolFormer — Distance-Stratified Softmax Attention × 3D Distance",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "molformer_exp1c_stratified.png", dpi=400, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(
        axes,
        ["cosine", "pearson", "spearman"],
        ["Cosine Similarity", "Pearson", "Spearman"],
    ):
        layers = np.arange(1, n_layers + 1)
        ax.errorbar(layers, corr[key].mean(1), yerr=corr[key].std(1),
                    marker="o", capsize=3, lw=2)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(f"Mean {title}", fontsize=12)
        ax.set_xticks(layers)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.suptitle("MolFormer — Mean Layerwise Softmax Attention × 3D Distance",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "molformer_exp1c_layerwise.png", dpi=400, bbox_inches="tight")
    plt.close()

    print("Experiment 1c plots saved.")


def main():
    csv_path = "./data/qm9.csv"

    print("=" * 65)
    print("Mechanistic Interpretability — MolFormer (Exp 1c: Softmax Attention)")
    print("=" * 65)
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"Device    : {DEVICE}")

    model, tokenizer, n_layers, n_heads = load_model()
    smiles = load_smiles(csv_path, n_samples=1000)

    print("\n" + "=" * 65)
    print("EXPERIMENT 1c: Softmax Attention × 3D Distance")
    print("=" * 65)

    raw = extract_attention_and_distances(
        model, tokenizer, smiles, n_layers, n_heads, max_molecules=1000)

    corr = compute_correlations(raw, n_layers, n_heads)
    plot_exp1c(corr, n_layers, n_heads)

    save_corr = {k: v.tolist() for k, v in corr.items()}
    with open(RESULTS_DIR / "molformer_exp1c_correlations.json", "w") as f:
        json.dump(save_corr, f, indent=2)

    print("\n" + "=" * 65)
    print("DONE")
    print(f"Results : {RESULTS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
