"""
Mechanistic Interpretability of SMI-TED — Experiment 1c.

Variant of smi_ted_exp1.py that uses the TRUE FAVOR+ effective attention
instead of a softmax proxy.  SMI-TED uses GeneralizedRandomFeatures with
kernel_fn=relu, so the actual attention weight is:

    A_ij^h = φ(q_i^h) · φ(k_j^h)
             ────────────────────────────────
             φ(q_i^h) · Σ_k φ(k_k^h)  + ε

where  φ(x) = relu(x @ omega)  and omega is the per-layer random projection
matrix saved in the checkpoint.  This is compared against 3D Euclidean
distance (same as exp1) so the two experiments are directly comparable.

Run from Desktop/smi_ted/:
    python smi_ted_exp1c.py [path/to/smiles.csv]
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

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "inference" / "smi_ted_light"
sys.path.insert(0, str(MODEL_DIR))

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

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATOM_PATTERN = re.compile(r"\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


def load_model():
    from load import load_smi_ted

    print(f"Loading SMI-TED from {MODEL_DIR}")
    model = load_smi_ted(
        folder=str(MODEL_DIR),
        ckpt_filename="smi-ted-Light_40.pt",
        vocab_filename="bert_vocab_curated.txt",
    )

    model.encoder.to(DEVICE)

    for layer in model.encoder.blocks.layers:
        layer.attention.inner_attention.feature_map.to(DEVICE)

    model.encoder.eval()

    cfg = model.config
    n_layers = cfg["n_layer"]
    n_heads = cfg["n_head"]

    print(f"Architecture: {n_layers} layers × {n_heads} heads | device: {DEVICE}")
    return model, n_layers, n_heads


def load_smiles(csv_path=None, n_samples=500):
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        smiles = df[col].dropna().tolist()
        print(f"Loaded {len(smiles)} SMILES from {csv_path}")
    else:
        print("No CSV supplied")
        smiles = []

    random.shuffle(smiles)
    return smiles[:n_samples]


def get_atom_map(smi, tokenizer):
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
        if ATOM_PATTERN.fullmatch(tok) and tok != "[H]":
            atom_map.append(cur)
            cur += 1
        else:
            atom_map.append(-1)

    full = [-1] + atom_map + [-1]
    return full, mol, smi


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
    heavy = [
        i for i in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(i).GetAtomicNum() != 1
    ]

    pos = np.array([list(conf.GetAtomPosition(i)) for i in heavy])
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def favor_effective_attention(queries, keys, feature_map):
    q = queries[0].float().to(DEVICE)
    k = keys[0].float().to(DEVICE)

    feature_map.to(DEVICE)

    q_feat = feature_map(q)
    k_feat = feature_map(k)

    num = torch.einsum("lhd,shd->lsh", q_feat, k_feat)

    k_sum = k_feat.sum(dim=0)
    denom = torch.einsum("lhd,hd->lh", q_feat, k_sum)

    eps = 1e-6
    A = num / (denom.unsqueeze(1) + eps)
    A = A.permute(2, 0, 1)

    A = A.clamp(min=0)
    row_sum = A.sum(dim=-1, keepdim=True).clamp(min=eps)
    A = A / row_sum

    return A.detach().cpu().numpy()


def extract_attention_and_distances(model, smiles_list, n_layers, n_heads, max_molecules=300):
    from fast_transformers.events import EventDispatcher, QKVEvent

    tokenizer = model.tokenizer
    enc = model.encoder
    dispatcher = EventDispatcher.get("")

    attn_modules = {
        enc.blocks.layers[i].attention: i for i in range(n_layers)
    }

    feature_maps = {
        i: enc.blocks.layers[i].attention.inner_attention.feature_map
        for i in range(n_layers)
    }

    qkv_store = {}

    def listener(event):
        if isinstance(event, QKVEvent) and event.source in attn_modules:
            layer_idx = attn_modules[event.source]
            qkv_store[layer_idx] = (
                event.queries.detach(),
                event.keys.detach(),
            )

    handle = dispatcher.listen(QKVEvent, listener)
    results = []

    for smi in tqdm(smiles_list, desc="Exp1c — attention extraction"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() < 3:
            continue

        dist_mat = get_3d_distance_matrix(mol)
        if dist_mat is None:
            continue

        full_map, mol_obj, raw_smi = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue

        a_idx = atom_token_indices(full_map)
        if len(a_idx) != dist_mat.shape[0]:
            continue

        idx, mask = model.tokenize(raw_smi)
        idx = idx.to(DEVICE)
        mask = mask.to(DEVICE)

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
            fm = feature_maps[layer]

            attn = favor_effective_attention(q, k, fm)
            sub = attn[:, a_idx, :][:, :, a_idx]
            atom_atts.append(sub)

        atom_atts = np.array(atom_atts)

        results.append({
            "smiles": smi,
            "dist_matrix": dist_mat,
            "atom_attentions": atom_atts,
            "num_atoms": n_atoms,
        })

        if len(results) >= max_molecules:
            break

    dispatcher.remove(handle)
    print(f"Processed {len(results)} molecules for Exp 1c")
    return results


def compute_correlations(results, n_layers, n_heads):
    arrays = {
        k: np.zeros((n_layers, n_heads))
        for k in ["cosine", "pearson", "spearman", 
                  "short_cosine", "medium_cosine", "long_cosine",
                  "short_pearson", "medium_pearson", "long_pearson",
                  "short_spearman", "medium_spearman", "long_spearman"]
    }
    counts = np.zeros((n_layers, n_heads))

    for res in tqdm(results, desc="Exp1c — correlations"):
        dist = res["dist_matrix"]
        att = res["atom_attentions"]
        n = res["num_atoms"]

        if n < 3:
            continue

        inv_dist = np.zeros_like(dist)
        inv_dist[dist > 0] = 1.0 / dist[dist > 0]

        idx = np.triu_indices(n, k=1)
        d_flat = dist[idx]
        inv_flat = inv_dist[idx]

        if len(d_flat) < 3:
            continue

        sm = d_flat <= 2.0
        med = (d_flat > 2.0) & (d_flat <= 4.0)
        lg = d_flat > 4.0

        for layer in range(min(n_layers, att.shape[0])):
            for head in range(min(n_heads, att.shape[1])):
                a_flat = att[layer, head, :n, :n][idx]

                # no variance -> denominator 0
                if a_flat.std() < 1e-10 or inv_flat.std() < 1e-10:
                    continue

                try:
                    arrays["cosine"][layer, head] += 1 - cosine_dist(a_flat, inv_flat)
                    arrays["pearson"][layer, head] += pearsonr(a_flat, inv_flat)[0]
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
    lt = list(range(1, n_layers + 1))
    ht = list(range(1, n_heads + 1))
    ann = n_layers <= 12 and n_heads <= 12

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, title in zip(
        axes,
        ["cosine", "pearson", "spearman"],
        ["Cosine Similarity", "Pearson", "Spearman"],
    ):
        sns.heatmap(
            corr[key],
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=ht,
            yticklabels=lt,
            annot=ann,
            fmt=".2f",
            annot_kws={"size": 7},
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)

    plt.suptitle("SMI-TED — FAVOR+ Attention × 3D Distance", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "smited_exp1c_heatmaps.png", dpi=400, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for ax, key, title in zip(
        axes.flatten(),
        ["short_cosine", "medium_cosine", "long_cosine",
         "short_pearson", "medium_pearson", "long_pearson",
         "short_spearman", "medium_spearman", "long_spearman"],
        ["Short Cosine (≤2 Å)", "Medium Cosine (2–4 Å)", "Long Cosine (>4 Å)",
         "Short Pearson (≤2 Å)", "Medium Pearson (2–4 Å)", "Long Pearson (>4 Å)",
         "Short Spearman (≤2 Å)", "Medium Spearman (2–4 Å)", "Long Spearman (>4 Å)"],
    ):
        sns.heatmap(
            corr[key],
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=ht,
            yticklabels=lt,
            annot=ann,
            fmt=".2f",
            annot_kws={"size": 7},
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)

    plt.suptitle("SMI-TED — Distance-Stratified FAVOR+ Attention × 3D Distance", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "smited_exp1c_stratified.png", dpi=400, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, title in zip(
        axes,
        ["cosine", "pearson", "spearman"],
        ["Cosine Similarity", "Pearson", "Spearman"],
    ):                                                                                           
        layers = np.arange(1, n_layers + 1)                                                                               
                                                                                                                            
        ax.errorbar(layers, corr[key].mean(1), yerr=corr[key].std(1), marker="o", capsize=3, lw=2)                                      
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel(f"Mean {title}", fontsize=12)                                                                     
        ax.set_xticks(layers)                                                                                             
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)  

    plt.suptitle("SMI-TED — Mean Layerwise FAVOR+ Attention × 3D Distance", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "smited_exp1c_layerwise.png", dpi=400, bbox_inches="tight")
    plt.close()

    print("Experiment 1c plots saved.")


def main():
    csv_path = "./data/qm9.csv"

    print("=" * 65)
    print("Mechanistic Interpretability — SMI-TED (Exp 1c: FAVOR+ attn)")
    print("=" * 65)
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"Device    : {DEVICE}")

    model, n_layers, n_heads = load_model()
    smiles = load_smiles(csv_path, n_samples=1000)

    print("\n" + "=" * 65)
    print("EXPERIMENT 1c: True FAVOR+ Attention × 3D Distance")
    print("=" * 65)

    exp1c_raw = extract_attention_and_distances(
        model,
        smiles,
        n_layers,
        n_heads,
        max_molecules=1000,
    )

    corr = compute_correlations(exp1c_raw, n_layers, n_heads)
    plot_exp1c(corr, n_layers, n_heads)

    save_corr = {k: v.tolist() for k, v in corr.items()}
    with open(RESULTS_DIR / "smited_exp1c_correlations.json", "w") as f:
        json.dump(save_corr, f, indent=2)


if __name__ == "__main__":
    main()