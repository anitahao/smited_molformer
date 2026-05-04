"""
Experiment 2: MOLFormer vs SMI-TED Comparison Plots
====================================================
Reads results from both models and produces one comparison
figure per property per dataset (QM9 and ESOL).

Run from smited_molformer/:
    python plot_comparison.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'comparison' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['qm9', 'esol']

PROPERTIES = [
    'atom_type',
    'hybridization',
    'is_aromatic',
    'is_in_ring',
    'chiral_tag',
    'degree',
    'formal_charge',
    'total_valence',
]


# ─────────────────────────────────────────────────────────────────────
# Load Results
# ─────────────────────────────────────────────────────────────────────

def load_results(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    return {
        task: {int(k): v for k, v in layer_res.items()}
        for task, layer_res in raw.items()
    }


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_comparison(molformer_results, smited_results,
                    dataset_name, fig_dir):
    """
    One figure per property. Each figure shows:
      - MOLFormer model activations   (solid blue)
      - SMI-TED model activations     (solid orange)
      - MOLFormer random baseline     (dashed blue)
      - SMI-TED random baseline       (dashed orange)
      - Frequency baseline            (dotted red)
    Unified y-axis (0 to 1).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    for prop in PROPERTIES:
        if prop not in molformer_results and prop not in smited_results:
            print(f"Skipping {prop} — not found in either model")
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        # MOLFormer
        if prop in molformer_results:
            mol_res    = molformer_results[prop]
            mol_layers = sorted(mol_res.keys())
            mol_accs   = [mol_res[l]['accuracy'] for l in mol_layers]
            ax.plot(mol_layers, mol_accs, marker='o', linewidth=2,
                    color='steelblue', label='MOLFormer')

            if 'accuracy_random' in mol_res[mol_layers[0]]:
                mol_rand = [mol_res[l]['accuracy_random']
                            for l in mol_layers]
                ax.plot(mol_layers, mol_rand, marker='s',
                        linewidth=1.5, linestyle='--',
                        color='steelblue', alpha=0.5,
                        label='MOLFormer (random input)')

            freq_acc = mol_res[mol_layers[0]]['frequency_baseline']
            ax.axhline(freq_acc, linestyle=':', linewidth=2,
                       color='tomato',
                       label=f'Frequency baseline ({freq_acc:.2f})')

        # SMI-TED
        if prop in smited_results:
            smi_res    = smited_results[prop]
            smi_layers = sorted(smi_res.keys())
            smi_accs   = [smi_res[l]['accuracy'] for l in smi_layers]
            ax.plot(smi_layers, smi_accs, marker='o', linewidth=2,
                    color='darkorange', label='SMI-TED')

            if 'accuracy_random' in smi_res[smi_layers[0]]:
                smi_rand = [smi_res[l]['accuracy_random']
                            for l in smi_layers]
                ax.plot(smi_layers, smi_rand, marker='s',
                        linewidth=1.5, linestyle='--',
                        color='darkorange', alpha=0.5,
                        label='SMI-TED (random input)')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(
            f'MOLFormer vs SMI-TED — {prop} ({dataset_name})',
            fontsize=13)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = fig_dir / f'exp2_comparison_{prop}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Experiment 2: MOLFormer vs SMI-TED Comparison")
    print("=" * 65)

    for dataset in DATASETS:
        print(f"\nDataset: {dataset.upper()}")

        mol_path = RESULTS_DIR / 'molformer' / f'exp2_probing_{dataset}.json'
        smi_path = RESULTS_DIR / 'smi_ted'   / f'exp2_probing_{dataset}.json'

        if not mol_path.exists():
            print(f"  MOLFormer results not found: {mol_path}")
            continue
        if not smi_path.exists():
            print(f"  SMI-TED results not found: {smi_path}")
            continue

        mol_results = load_results(mol_path)
        smi_results = load_results(smi_path)

        print(f"  MOLFormer properties: {list(mol_results.keys())}")
        print(f"  SMI-TED   properties: {list(smi_results.keys())}")

        fig_dir = FIGURES_DIR / dataset
        plot_comparison(mol_results, smi_results,
                        dataset.upper(), fig_dir)

    print("\n" + "=" * 65)
    print("COMPARISON PLOTS COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 65)


if __name__ == '__main__':
    main()
