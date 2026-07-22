# CoMSES Net deposit checklist

Use this list when uploading Dynamic-BABE to the
[CoMSES Computational Model Library](https://www.comses.net/).

## Required content

- [ ] Source code: `agent.py`, `model.py`, `config.py`, `batch_run.py`, `stats_analysis.py`, `visualize.py`, `sensitivity.py`, `experiments/`
- [ ] `requirements.txt` and `LICENSE` (MIT)
- [ ] `README.md` (overview, seeds, pipeline)
- [ ] ODD+D description: `paper/sections/04_method_odd.tex` (or exported PDF section)
- [ ] Default parameters: `config.py`
- [ ] Example outputs: `output/statistical_analysis.csv`, `output/sensitivity_results.csv` (optional but helpful)
- [ ] Figures or figure-generation script: `visualize.py`

## Access for review

1. Create / confirm CoMSES membership.
2. Upload a release zip or sync from GitHub (`https://github.com/TheCoolSam/BABEModel`).
3. Set reviewer access if the journal requests a private handle before publication.
4. After acceptance, publish so a permanent handle can be minted.
5. Paste the CoMSES URL into:
   - `paper/main.tex` (Code and Data Availability)
   - `paper/main_blind.tex` (neutral wording + handle if non-identifying)
   - `paper/sections/02_introduction.tex` (replication sentence / endnote)

## Author bios (separate from CoMSES)

Bios and corresponding email are still author-supplied — see `paper/SUBMISSION_GATES.md`.
