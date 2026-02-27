"""
config.py — Dynamic-BABE Model Global Configuration

All simulation constants centralised here for reproducibility.
Change values here; never hard-code numbers in agent.py / model.py.

References
----------
* Chen, W., Pacheco, D., Yang, K.-C., & Menczer, F. (2021).
  "Neutral bots probe political bias on social media." *Nature Communications*.
* Jager, W., & Amblard, F. (2005).
  "Uniformity, bipolarization and pluriformity captured as generic stylized
  behavior with an agent-based simulation model of attitude change."
  *Computational & Mathematical Organization Theory*.
* DeGroot, M. H. (1974). "Reaching a consensus."
  *Journal of the American Statistical Association*.
"""

# ──────────────────────────────────────────────────────────────────
# 1. POPULATION & NETWORK TOPOLOGY
# ──────────────────────────────────────────────────────────────────
NUM_AGENTS = 200              # Network node count
BA_EDGE_PARAM = 3             # m in Barabasi–Albert G(n, m)
NUM_ISSUES = 2                # Dimensionality of opinion vector
                                   # (0 = Economic, 1 = Social)
OPINION_INIT_MODE = "bipolar"  # "uniform" = U(-1,1);  "bipolar" = two camps
                               # Bipolar models an already-divided society

# ──────────────────────────────────────────────────────────────────
# 2. AGENT COGNITIVE PARAMETERS
# ──────────────────────────────────────────────────────────────────
BETA_MEAN = 4.0             # Mean cognitive entrenchment (β_i)
                             # ↑ Models hyper-partisan population (Phase Transition regime)
BETA_STD = 1.5              # Std-dev for β distribution (wider spread = more extremists)
BETA_MIN = 0.5              # Floor clamp – nobody is perfectly open
INTERACTIONS_PER_STEP = 1     # Neighbours sampled per tick

# ──────────────────────────────────────────────────────────────────
# 3. SOCIAL-JUDGMENT THRESHOLDS  (Jager-Amblard Zones)
# ──────────────────────────────────────────────────────────────────
ASSIMILATION_THRESHOLD = 0.3   # w_ij > this → Zone 1 (accept)
# Zone 2 (non-commitment):  0  ≤ w_ij ≤ ASSIMILATION_THRESHOLD
# Zone 3 (backfire):         w_ij < 0

# ──────────────────────────────────────────────────────────────────
# 4. NOVEL CONTRIBUTION — TOXIC CHURN HYPOTHESIS
# ──────────────────────────────────────────────────────────────────
CHURN_THRESHOLD = 15          # T_c: frustration ceiling before churn
HEALING_RATE = 1              # Frustration healed per assimilation event
                              # Models positive social interactions restoring faith
BRIDGE_HEALING_BONUS = 2      # Extra frustration healed when Bridge intercepts
                              # a backfire — successful moderation rebuilds trust

# ──────────────────────────────────────────────────────────────────
# 5. REVENUE STEP-FUNCTION
# ──────────────────────────────────────────────────────────────────
BASE_AD_RATE = 1.0          # Revenue-per-active-user (normalised $)
POLARIZATION_CLIFF = 0.6   # σ threshold for Brand Safety Penalty
SAFE_AD_MULTIPLIER = 1.0   # Ad rate when σ < cliff (100 %)
UNSAFE_AD_MULTIPLIER = 0.4 # Ad rate when σ ≥ cliff (40 %)

# ──────────────────────────────────────────────────────────────────
# 6. THE BRIDGE ALGORITHM  (Community-Notes–inspired dampener)
# ──────────────────────────────────────────────────────────────────
ENABLE_BRIDGE = False        # Toggle for A/B experiments
BRIDGE_EFFICACY = 0.46     # 46 % success rate per intercept

# ──────────────────────────────────────────────────────────────────
# 7. SIMULATION CONTROL
# ──────────────────────────────────────────────────────────────────
MAX_STEPS = 200               # Steps per single run
BATCH_ITERATIONS = 30         # Runs for batch_run.py (30 = sufficient for CI)
RANDOM_SEED = 42              # Master seed for reproducibility
