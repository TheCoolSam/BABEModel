# The Polarization Paradox

> **Theoretical Agent-Based Model of Algorithmic Moderation and Dyadic Trust Co-Evolution**
>
> *Target Journal:* **Journal of Artificial Societies and Social Simulation (JASSS)**
>
> *Authors:* Botshtein, Ghose, Pipal (Independent Researchers) & Ehlinger (University of Wisconsin–Milwaukee)
>
> *Code:* https://github.com/TheCoolSam/BABEModel

---

## Research Overview

This repository implements **Dynamic-BABE (Biased Assimilation & Behavioral Entrenchment)**, an agent-based model of how bridging-style feed interventions interact with co-evolving dyadic trust.

**Polarization Paradox:** bridging sharply reduces churn and protects step-wise revenue, yet final polarization can rise because extreme agents who would otherwise exit remain active.

**Trust segregation:** enabling trust co-evolution (Trust Only vs Baseline) concentrates trust inside ideological silos while outgroup trust collapses. That contrast is distinct from the bridging paradox.

---

## Reproducibility (seeds)

- Base seed: `RANDOM_SEED = 42` in `config.py`
- Batch run \(i\) uses seed `42 + i` (paired across the four factorial conditions)
- Partner choice / bridge draws: `numpy.random.Generator(seed)`
- Mesa `RandomActivation` shuffle: Mesa 2.x `model.random` via `self._seed` set before the scheduler is created

---

## Setup & Pipeline

```powershell
cd path\to\polarizationParadox
python -m venv .venv
.\.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python batch_run.py               # 4 conditions × 30 runs (long)
python stats_analysis.py          # Mann–Whitney pairwise + Wilcoxon interactions
python visualize.py               # figures/ PNGs (~800 px wide for JASSS)
python sensitivity.py             # OFAT robustness (long); --quick for smoke test
```

Compile the paper from the `paper/` directory so `\graphicspath{{../}}` resolves `figures/`:

```powershell
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 2×2 Factorial Design

| Condition | Bridge | Trust | CSV suffix |
|-----------|--------|-------|------------|
| Baseline | OFF | OFF | `_baseline` |
| Bridge Only | ON | OFF | `_bridge_only` |
| Trust Only | OFF | ON | `_trust_only` |
| Full Model | ON | ON | `_full_model` |

Canonical outputs: `output/model_data_*.csv`, `agent_data_*.csv`, `statistical_analysis.csv`, `interaction_analysis.csv`.

Legacy pre-factorial `bridge_on/off` dumps (if present) live under `output/legacy/` and are **not** part of the published design.

---

## Sensitivity (laptop-friendly)

Background OFAT without melting the machine (2 workers, 5 reps):

```powershell
$env:PYTHONUNBUFFERED="1"
.\.venv\Scripts\python.exe -u sensitivity.py --laptop
```

Optional knobs: `--workers=2` `--iters=5` (or `--quick` for 3 reps).
Default sensitivity parallelism is capped at 2 cores so Chrome/Cursor stay usable.


Before or upon acceptance, deposit a release on [CoMSES Net](https://www.comses.net/):

1. Create / confirm CoMSES membership.
2. Upload this repository (or a release zip) with `README.md`, `requirements.txt`, ODD+D paper section, and `config.py` defaults.
3. Set access for reviewers if needed; include the CoMSES URL/handle in the manuscript.
4. After acceptance, publish the model so a permanent handle can be minted.
5. Keep GitHub (`https://github.com/TheCoolSam/BABEModel`) as the working mirror.

---

## License

MIT License. Open science: seed-reproducible batch design; analysis and figure scripts included.
