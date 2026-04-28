"""
Mechanistic Interpretability of MolFormer — Experiment 2.

Linear probing for chemical properties at each encoder layer.

MolFormer uses standard softmax attention and the HuggingFace API,
so hidden states are retrieved via output_hidden_states=True (no hooks needed).

Run from Desktop/molformer/:
    python molformer_exp2.py
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
FIGURES_DIR = BASE_DIR / "figures" / "exp2"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    max_len  = getattr(model.config, "max_position_embeddings", 202)
    print(f"Architecture: {n_layers} layers × {n_heads} heads  |  device: {DEVICE}")
    return model, tokenizer, n_layers, n_heads, max_len


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
    n   = mol.GetNumAtoms()

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


def get_atom_properties(mol):
    return [{
        "atom_type":     a.GetSymbol(),
        "hybridization": str(a.GetHybridization()),
        "is_aromatic":   a.GetIsAromatic(),
        "is_in_ring":    a.IsInRing(),
        "chiral_tag":    str(a.GetChiralTag()),
        "degree":        a.GetDegree(),
        "formal_charge": a.GetFormalCharge(),
        "total_valence": a.GetTotalValence(),
    } for a in mol.GetAtoms()]


def extract_hidden_states(model, tokenizer, smiles_list, n_layers, max_molecules=500):
    # MolFormer returns all hidden states via output_hidden_states=True.
    # outputs.hidden_states: tuple of length n_layers+1
    #   [0] = embedding output, [1..n_layers] = transformer layer outputs
    # Each tensor: (1, seq_len, d_model)

    layer_out   = {i: [] for i in range(n_layers + 1)}
    atom_labels = []
    processed   = 0

    for smi in tqdm(smiles_list, desc="Exp2 — hidden states"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        full_map, mol_obj = get_atom_map(smi, tokenizer)
        if full_map is None:
            continue

        a_idx = atom_token_indices(full_map)
        if len(a_idx) != mol.GetNumAtoms():
            continue

        inputs = tokenizer(smi, return_tensors="pt", padding=False,
                           truncation=True, max_length=202)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        seq_len = inputs["input_ids"].shape[1]
        a_idx_valid = [i for i in a_idx if i < seq_len]
        if len(a_idx_valid) != mol.GetNumAtoms():
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (1, seq_len, d_model)
        for layer_idx, hs in enumerate(outputs.hidden_states):
            atom_hs = hs[0][a_idx_valid].cpu().numpy()   # (n_atoms, d_model)
            layer_out[layer_idx].append(atom_hs)

        for prop in get_atom_properties(mol_obj):
            atom_labels.append({**prop, "mol_id": processed})

        processed += 1
        if processed >= max_molecules:
            break

    layer_emb = {
        i: np.concatenate(arrs, axis=0)
        for i, arrs in layer_out.items() if arrs
    }
    print(f"Extracted hidden states: {processed} molecules, {len(atom_labels)} atoms")
    return layer_emb, atom_labels


def train_test(atom_labels):
    df      = pd.DataFrame(atom_labels)
    mol_ids = df["mol_id"].values
    unique_mols = np.unique(mol_ids)
    np.random.shuffle(unique_mols)

    split_mol   = int(0.8 * len(unique_mols))
    train_mols  = set(unique_mols[:split_mol])
    tr = np.where( np.isin(mol_ids, list(train_mols)))[0]
    te = np.where(~np.isin(mol_ids, list(train_mols)))[0]
    return tr, te


def run_linear_probing(layer_emb, atom_labels, tr, te):
    df = pd.DataFrame(atom_labels)
    tasks = {
        "atom_type":     df["atom_type"].values,
        "hybridization": df["hybridization"].astype(str).values,
        "is_aromatic":   df["is_aromatic"].astype(int).values,
        "is_in_ring":    df["is_in_ring"].astype(int).values,
        "chiral_tag":    df["chiral_tag"].astype(str).values,
        "degree":        df["degree"].values,
        "formal_charge": df["formal_charge"].values,
        "total_valence": df["total_valence"].values,
    }

    results = {}

    for task, labels in tasks.items():
        print(f"\nProbing: {task}")
        le = LabelEncoder()
        y  = le.fit_transform(labels)
        print(f"  Classes: {le.classes_}")
        task_res = {}
        for layer_idx in sorted(layer_emb.keys()):
            X = layer_emb[layer_idx]
            if len(X) != len(y):
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegressionCV(max_iter=1000, random_state=SEED,
                                                     n_jobs=-1, C=1.0))
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            acc  = accuracy_score(y[te], pred)
            f1   = f1_score(y[te], pred, average="weighted")
            task_res[layer_idx] = {"accuracy": acc, "f1": f1}
            print(f"  Layer {layer_idx}: Acc={acc:.4f}  F1={f1:.4f}")
        results[task] = task_res

    return results


def run_dummy_classifier(atom_labels, tr, te):
    df = pd.DataFrame(atom_labels)
    tasks = {
        "atom_type":     df["atom_type"].values,
        "hybridization": df["hybridization"].astype(str).values,
        "is_aromatic":   df["is_aromatic"].astype(int).values,
        "is_in_ring":    df["is_in_ring"].astype(int).values,
        "chiral_tag":    df["chiral_tag"].astype(str).values,
        "degree":        df["degree"].values,
        "formal_charge": df["formal_charge"].values,
        "total_valence": df["total_valence"].values,
    }
    results = {}
    for task, label in tasks.items():
        print(f"\nDummyClassifier: {task}")
        le  = LabelEncoder()
        y   = le.fit_transform(label)
        dum = DummyClassifier(strategy="most_frequent")
        dum.fit(np.zeros((len(tr), 1)), y[tr])
        acc = accuracy_score(y[te], dum.predict(np.zeros((len(te), 1))))
        results[task] = {"accuracy": acc}
        print(f"  {task}: baseline acc = {acc:.4f}")
    return results


def plot_exp2(probing_results, dummy_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for task, res in probing_results.items():
        layers = sorted(res.keys())
        line, = axes[0].plot(layers, [res[l]["accuracy"] for l in layers],
                             marker="o", label=task, lw=2)
        axes[1].plot(layers, [res[l]["f1"] for l in layers],
                     marker="o", label=task, lw=2, color=line.get_color())
        baseline = dummy_results[task]["accuracy"]
        axes[0].axhline(baseline, ls="--", color=line.get_color(), alpha=0.5)
        axes[1].axhline(baseline, ls="--", color=line.get_color(), alpha=0.5)
    for ax, ylabel, title in zip(
        axes,
        ["Accuracy", "F1 (weighted)"],
        ["Probe Accuracy by Layer", "Probe F1 by Layer"],
    ):
        ax.set_xlabel("Layer", fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"MolFormer — {title}", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "molformer_exp2_probing.png", dpi=400, bbox_inches="tight")
    plt.close()
    print("Experiment 2 plots saved.")


def main():
    csv_path = "./data/qm9.csv"

    print("=" * 65)
    print("Mechanistic Interpretability — MolFormer")
    print("=" * 65)
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"Device    : {DEVICE}")

    model, tokenizer, n_layers, n_heads, max_len = load_model()
    smiles = load_smiles(csv_path, n_samples=1000, max_len=150)

    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Linear Probing for Chemical Properties")
    print("=" * 65)

    layer_emb, atom_labels = extract_hidden_states(
        model, tokenizer, smiles, n_layers, max_molecules=1000)

    tr, te     = train_test(atom_labels)
    probe_res  = run_linear_probing(layer_emb, atom_labels, tr, te)
    dummy_res  = run_dummy_classifier(atom_labels, tr, te)
    plot_exp2(probe_res, dummy_res)

    with open(RESULTS_DIR / "molformer_exp2_probing.json", "w") as f:
        json.dump({task: {str(k): v for k, v in res.items()}
                   for task, res in probe_res.items()}, f, indent=2)

    print("\n" + "=" * 65)
    print("DONE")
    print(f"Results : {RESULTS_DIR}")
    print(f"Figures : {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
