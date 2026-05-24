# The Polarization Paradox

> **Theoretical Agent-Based Model of Algorithmic Moderation and Dyadic Trust Co-Evolution**
> 
> *Target Journal:* **Journal of Artificial Societies and Social Simulation (JASSS)**
> 
> *Authors:* Botshtein, Ghose, Pipal (Independent Researchers) & Dr. Ehlinger (University of Wisconsin-Milwaukee)

---

## 🌟 Research Overview

This repository contains the complete implementation of the **Dynamic-BABE (Biased Assimilation & Behavioral Entrenchment) Model**, an agent-based simulation exploring how algorithmic content moderation interacts with the co-evolution of dyadic interpersonal trust.

### 🎭 The Paradox

In modern social media platforms, toxic polarization drives user churn, hurting platform revenue. To counter this, platforms deploy **bridging algorithms** that suppress "backfire" (rejection-zone) interactions. 

However, we discover a paradoxical outcome: **bridging algorithms preserve platform revenue and user retention but fail to mitigate systemic polarization**. When dyadic trust is fragile, suppressing confrontational cross-cutting interactions deprives agents of the opportunity to build mutual tolerance. As a result, the network experiences **trust segregation**—where trust is concentrated strictly within ideological echo chambers—ultimately entrenching social division.

---

## 📐 Model Architecture (ODD+D Overview)

Our model extends the classic **DeGroot consensus model** by incorporating cognitive psychology and network science:

1. **State-Vector Agents**: Each agent $i$ possesses:
   - An $N$-dimensional opinion vector $\vec{O}_i \in [-1, 1]^N$.
   - A salience vector $\vec{S}_i$ (Dirichlet distributed) weighting issue importance.
   - An entrenchment parameter $\beta_i$ (cognitive resistance).
   - An active flag (agents churn permanently if frustration exceeds a threshold $T_c$).

2. **Dyadic Trust Co-evolution**: Every edge $(i, j)$ stores a symmetric trust weight $T_{ij} \in [0, 1]$.
   - **Buffer Effect**: Friends can disagree. High trust buffers agents, shifting influence weights upward to prevent backfiring.
   - **Asymmetric Co-evolution**: Trust builds slowly upon agreement ($\delta_+$) but erodes rapidly upon backfire conflict ($\delta_-$).
   - **Passive Decay**: Relationships wither over time ($\lambda$ decay) if not actively maintained.

3. **Bridge Algorithm**: When an interaction falls into the Latitude of Rejection (Zone 3), the algorithm intervenes with efficacy $E = 46\%$, neutralizing the backfire and preventing frustration.

---

## 📁 Repository Structure

```
├── agent.py            # State-vector SocialAgent (maths, assimilation, backfire)
├── model.py            # SocialNetworkModel (BA graph, interaction loop, data collectors)
├── config.py           # Centralized simulation hyperparameters
├── batch_run.py        # Parallel batch runner for 2×2 factorial design
├── stats_analysis.py   # Pairwise comparisons & Bridge×Trust interaction analysis
├── visualize.py        # Generates 10+ publication-quality figures
├── requirements.txt    # Python package dependencies
└── output/             # Exported CSVs and statistical tables
```

---

## 🖥️ Setup & Run Instructions (macOS & Windows)

### 🍏 For macOS Users (MacBook Air/Pro M1/M2/M3/M4)

You do **not** need a specialized code editor to run the model, though we highly recommend installing **[VS Code](https://code.visualstudio.com/)** to inspect the code, configuration, and results.

Follow these step-by-step terminal commands to set up and run the code:

#### Step 1: Open the Terminal
Press `Cmd + Space` to open Spotlight, type **Terminal**, and press `Enter`.

#### Step 2: Navigate to your repository
Navigate to the directory where you cloned the code (replace the path with your actual folder location):
```bash
cd /path/to/polarizationParadox
```

#### Step 3: Create a clean Python Virtual Environment
macOS requires virtual environments to avoid messing with system packages. Create one using:
```bash
python3 -m venv .venv
```

#### Step 4: Activate the environment
```bash
source .venv/bin/activate
```
*(Your terminal prompt should now show `(.venv)` at the beginning).*

#### Step 5: Upgrade pip and install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 6: Run the simulation pipeline
Run the parallel 2×2 factorial batch simulation (executes 120 runs across all CPU cores):
```bash
python batch_run.py
```
*Outputs will be saved as 8 CSV files in the `./output/` directory.*

#### Step 7: Run statistical analysis
Execute pairwise Mann-Whitney U, Cohen's d, and seed-paired Bridge×Trust interaction tests:
```bash
python stats_analysis.py
```
*Outputs formatted tables to the terminal and saves CSV summaries in `./output/`.*

#### Step 8: Generate publication-ready figures
```bash
python visualize.py
```
*Outputs 14 publication-quality PNG figures in the `./figures/` directory.*

---

### 🪟 For Windows Users

1. Open **PowerShell** or **Command Prompt**.
2. Navigate to your folder: `cd C:\path\to\polarizationParadox`
3. Create venv: `python -m venv .venv`
4. Activate venv: `.venv\Scripts\activate`
5. Install: `pip install -r requirements.txt`
6. Run:
   ```powershell
   python batch_run.py
   python stats_analysis.py
   python visualize.py
   ```

---

## 📊 2×2 Factorial Design & Figures

Our experiment executes a **2×2 Factorial Design** crossing the presence of the Bridge Algorithm and Dyadic Trust:

| Condition Name | Bridge Algorithm | Dyadic Trust | CSV Suffix | Color |
|----------------|------------------|--------------|------------|-------|
| **Baseline** | OFF | OFF | `_baseline` | 🔴 Red |
| **Bridge Only**| ON | OFF | `_bridge_only`| 🔵 Blue |
| **Trust Only** | OFF | ON | `_trust_only` | 🟢 Green |
| **Full Model** | ON | ON | `_full_model` | 🟣 Purple |

### Generated Figures Manifest

All figures are compiled directly into publication-ready sub-files inside `/figures/`:

*   **`fig2_polarization_factorial.png`**: Multi-run average polarization over time with 95% Confidence Intervals.
*   **`fig3_churn_factorial.png`**: System-level cumulative user churn rates.
*   **`fig4_revenue_factorial.png`**: Platform revenue dynamics showing the "polarization cliff" penalty.
*   **`fig5_summary_barplot.png`**: A 5-panel final-step comparison of all KPIs with error bars.
*   **`fig6_trust_dynamics.png`**: Network-wide trust evolution and ingroup/outgroup trust divergence.
*   **`fig7_echo_chambers.png`**: Opinion-weighted clustering coefficient measuring echo chamber entrenchment.
*   **`figS1` to `figS6`**: Supplementary figures for frustration, active user retention, opinion extremity, and trust segregation ratios.

---

## 📄 License & Open Science

This model is fully open-source under the MIT License. In compliance with **JASSS Open Science guidelines**, all code is fully seed-reproducible. The master seed is locked in `config.py` as `RANDOM_SEED = 42`.
