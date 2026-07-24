# The Polarization Paradox

Theoretical agent-based model of a suppression-oriented feed filter and dyadic trust co-evolution.

- Target journal: Journal of Artificial Societies and Social Simulation (JASSS)
- Authors: Botshtein, Ghose, Pipal (independent researchers); Ehlinger (University of Wisconsin–Milwaukee)
- Code: https://github.com/TheCoolSam/BABEModel

## Overview

Dynamic-BABE (Biased Assimilation and Behavioral Entrenchment) combines biased assimilation, social-judgment zones, exit via frustration, and optional co-evolving dyadic trust.

Main result (Polarization Paradox): turning the suppression filter on sharply cuts churn and protects step-wise revenue, while active-user polarization stays high or rises because extreme agents who would otherwise exit remain active. Full-cohort polarization (including churned agents' last opinions) does not show that rise.

Trust segregation is a separate contrast: enabling trust co-evolution concentrates trust within opinion camps.

Manuscript term: suppression-oriented feed filter (short: suppression filter). Code flag remains `enable_bridge` for API continuity. This is not bridging-based ranking that promotes cross-camp content.

## Install and run

```bash
git clone https://github.com/TheCoolSam/BABEModel.git
cd BABEModel

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt

python scripts/check_reproducibility.py
python batch_run.py --workers=2
python stats_analysis.py
python visualize.py
```

Optional:

```bash
python sensitivity.py --laptop
python -m experiments.ablations --laptop --iters=20
python -m experiments.baselines --laptop --iters=20
python -m experiments.topology_robustness --laptop --iters=15
python scripts/mu_overshoot_diagnostic.py
```

Compile the paper from `paper/` so figures resolve via `\graphicspath{{../}}`:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Reproducibility

- Base seed: `RANDOM_SEED = 42` in `config.py`
- Batch run `i` uses seed `42 + i` (paired across the four factorial cells)
- Partner choice and filter draws: `numpy.random.Generator(seed)`
- Mesa `RandomActivation` shuffle: `model.reset_randomizer(seed)`

## 2x2 factorial design

| Condition   | Filter | Trust | CSV suffix      |
|-------------|--------|-------|-----------------|
| Baseline    | OFF    | OFF   | `_baseline`     |
| Filter Only | ON     | OFF   | `_bridge_only`  |
| Trust Only  | OFF    | ON    | `_trust_only`   |
| Full Model  | ON     | ON    | `_full_model`   |

Filter ON/OFF maps to `enable_bridge=True/False`. CSV suffixes keep `_bridge_only` for legacy filenames.

Canonical outputs: `output/model_data_*.csv`, `agent_data_*.csv`, `statistical_analysis.csv`, `interaction_analysis.csv`.

## Sensitivity

```bash
python sensitivity.py --laptop
```

Options: `--workers=2`, `--iters=5`, or `--quick` for a smoke test.

## Additional experiments

- Colab: `notebooks/colab_run_experiments.ipynb`
- SSH: `bash scripts/remote_tmux.sh 20 4` then `tmux attach -t babe_experiments`
- Checkpoints: `output/checkpoints/*_partial.csv`

CoMSES deposit steps: `docs/COMSES_DEPOSIT.md`.

## License

MIT.
