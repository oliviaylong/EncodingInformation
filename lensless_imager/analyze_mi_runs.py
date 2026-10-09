"""
Summarize the MI estimates saved by 2024_10_23_pixelcnn_cifar10_updated_api_reruns_smaller_lr.py.

Reads every pixelcnn_mi_estimate_{photons}_photon_count_{psf}_psf.npy (and the matching val_loss file)
in a results folder, prints a table, checks the expected trends, and saves plots to that folder.

Usage:
    python analyze_mi_runs.py [results_dir]
"""

import os
import re
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on the GPU server; save plots to files instead
import matplotlib.pyplot as plt

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # running in a Jupyter / VS Code interactive window
    HERE = os.getcwd()
DEFAULT_DIR = os.path.join(HERE, "mi_estimates_smaller_lr", "check")
# less multiplexing -> more information for dense CIFAR-10 images
PSF_ORDER = ["one", "two", "three", "four", "five", "six", "seven", "diffuser"]
PATTERN = re.compile(r"pixelcnn_mi_estimate_(\d+)_photon_count_(.+)_psf\.npy$")


def load_results(results_dir):
    """Return {(photons, psf): {"mi", "lo", "hi", "val_losses"}}, one array entry per seed."""
    results = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "pixelcnn_mi_estimate_*_psf.npy"))):
        match = PATTERN.search(os.path.basename(path))
        if match is None:
            continue
        photons, psf = int(match.group(1)), match.group(2)
        mi, lo, hi = np.atleast_2d(np.load(path))[:3]
        val_path = path.replace("pixelcnn_mi_estimate_", "pixelcnn_val_loss_")
        val_losses = list(np.load(val_path, allow_pickle=True)) if os.path.exists(val_path) else []
        results[(photons, psf)] = {"mi": mi, "lo": lo, "hi": hi, "val_losses": val_losses}
    return results


def psf_sort_key(psf):
    return PSF_ORDER.index(psf) if psf in PSF_ORDER else len(PSF_ORDER)


def print_table(results):
    print(f"{'photons':>8s}  {'psf':<10s} {'seed':>4s}  {'MI (bits/px)':>12s}  {'95% CI':>17s}  {'epochs':>6s}  {'final val NLL':>13s}")
    for (photons, psf) in sorted(results, key=lambda k: (k[0], psf_sort_key(k[1]))):
        r = results[(photons, psf)]
        for s in range(len(r["mi"])):
            curve = r["val_losses"][s] if s < len(r["val_losses"]) else []
            epochs = f"{len(curve) - 1}" if len(curve) else "-"
            final = f"{min(curve):.3f}" if len(curve) else "-"
            print(f"{photons:8d}  {psf:<10s} {s + 1:4d}  {r['mi'][s]:12.3f}  "
                  f"[{r['lo'][s]:.3f}, {r['hi'][s]:.3f}]  {epochs:>6s}  {final:>13s}")


def check_trends(results):
    """Print PASS/CHECK for the two trends the check runs are meant to show."""
    mean_mi = {k: float(np.mean(r["mi"])) for k, r in results.items()}
    print("\nTrend checks")

    # 1. At each photon count with several PSFs: less multiplexing should give more MI
    for photons in sorted({p for p, _ in results}):
        psfs = sorted([psf for p, psf in results if p == photons], key=psf_sort_key)
        if len(psfs) < 2:
            continue
        values = [mean_mi[(photons, psf)] for psf in psfs]
        ok = all(a > b for a, b in zip(values, values[1:]))
        order = " > ".join(f"{psf} ({v:.3f})" for psf, v in zip(psfs, values))
        print(f"  [{'PASS' if ok else 'CHECK'}] {photons} photons, MI decreases with multiplexing: {order}")

    # 2. For each PSF with several photon counts: more photons should give more MI
    for psf in sorted({psf for _, psf in results}, key=psf_sort_key):
        photon_list = sorted(p for p, s in results if s == psf)
        if len(photon_list) < 2:
            continue
        values = [mean_mi[(p, psf)] for p in photon_list]
        ok = all(a < b for a, b in zip(values, values[1:]))
        order = " < ".join(f"{p} ({v:.3f})" for p, v in zip(photon_list, values))
        print(f"  [{'PASS' if ok else 'CHECK'}] {psf} PSF, MI increases with photons: {order}")

    # 3. Did training converge? The curve should have flattened before early stopping.
    for (photons, psf), r in sorted(results.items()):
        for s, curve in enumerate(r["val_losses"]):
            curve = np.asarray(curve, dtype=float)
            if len(curve) < 25:
                print(f"  [CHECK] {photons} photons, {psf}, seed {s + 1}: only {len(curve) - 1} epochs, "
                      f"may not have converged")


def plot_results(results, results_dir):
    psfs = sorted({psf for _, psf in results}, key=psf_sort_key)
    photon_counts = sorted({p for p, _ in results})

    # MI vs photon count, one line per PSF (error bars: mean of the per-seed 95% CI bounds)
    fig, ax = plt.subplots(figsize=(6, 4))
    for psf in psfs:
        pts = sorted((p, results[(p, psf)]) for p in photon_counts if (p, psf) in results)
        x = [p for p, _ in pts]
        y = [np.mean(r["mi"]) for _, r in pts]
        yerr = [[np.mean(r["mi"]) - np.mean(r["lo"]) for _, r in pts],
                [np.mean(r["hi"]) - np.mean(r["mi"]) for _, r in pts]]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=psf)
    ax.set_xscale("log")
    ax.minorticks_off()  # otherwise log-axis minor labels overlap the photon-count labels
    ax.set_xticks(photon_counts)
    ax.set_xticklabels([str(p) for p in photon_counts])
    ax.set_xlabel("mean photons per pixel")
    ax.set_ylabel("MI (bits per pixel)")
    ax.legend(title="PSF")
    fig.tight_layout()
    mi_path = os.path.join(results_dir, "mi_vs_photons.png")
    fig.savefig(mi_path, dpi=150)
    plt.close(fig)

    # validation NLL curves, one panel per run
    runs = sorted(results.items(), key=lambda kv: (kv[0][0], psf_sort_key(kv[0][1])))
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3), squeeze=False)
    for ax, ((photons, psf), r) in zip(axes[0], runs):
        for s, curve in enumerate(r["val_losses"]):
            # skip epoch 0 (the untrained model), which dwarfs the rest of the curve
            ax.plot(range(1, len(curve)), np.asarray(curve, dtype=float)[1:], label=f"seed {s + 1}")
        ax.set_title(f"{psf}, {photons} photons", fontsize=9)
        ax.set_xlabel("epoch")
    axes[0][0].set_ylabel("validation NLL")
    fig.tight_layout()
    loss_path = os.path.join(results_dir, "val_loss_curves.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved {mi_path}\nSaved {loss_path}")


if __name__ == "__main__":
    # a folder given on the command line; interactive windows pass their own non-folder arguments, so skip those
    results_dir = next((a for a in sys.argv[1:] if os.path.isdir(a)), DEFAULT_DIR)
    results = load_results(results_dir)
    if not results:
        sys.exit(f"No pixelcnn_mi_estimate_*_psf.npy files found in {results_dir}\n"
                 f"Pass the folder that holds them: python analyze_mi_runs.py <results_dir>")
    print(f"Results in {results_dir}\n")
    print_table(results)
    check_trends(results)
    plot_results(results, results_dir)
