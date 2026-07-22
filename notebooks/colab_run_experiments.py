"""
Google Colab runner for Dynamic-BABE strength-upgrade experiments.

How to use
----------
1. Upload this repo (or clone from GitHub) into Colab / Drive.
2. Runtime → change runtime type (CPU is fine; more cores help).
3. Run all cells. Results checkpoint to Drive so disconnects lose little work.
4. Free Colab can still interrupt idle sessions — Colab Pro or SSH/tmux is safer
   for multi-hour runs. Closing *your laptop* is fine while Colab runs in the cloud.

You can also run the same CLIs in a Colab terminal:
  !python -m experiments.ablations --iters=20 --workers=2
"""

# --- Cell 1: setup ---
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/polarizationParadox   # or wherever you put the repo
# !pip install -q -r requirements.txt

# --- Cell 2: ablations ---
# !python -m experiments.ablations --iters=20 --workers=2 --outdir=output

# --- Cell 3: baselines ---
# !python -m experiments.baselines --iters=20 --workers=2 --outdir=output

# --- Cell 4: topology ---
# !python -m experiments.topology_robustness --iters=15 --workers=2 --outdir=output

# --- Cell 5: sync note ---
# Checkpoints land in output/checkpoints/*.csv and final files in output/*.csv
print("Open notebooks/colab_run_experiments.ipynb in Colab and run the cells.")
